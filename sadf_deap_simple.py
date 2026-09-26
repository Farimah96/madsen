"""
Multi-Objective Hardware/Software Co-Design for SADF Applications using NSGA-II
================================================================================

Simple SADF implementation:
    - Each scenario is an independent SDF graph.
    - Each scenario has its own rates_matrix.
    - Architecture allocation AND binding are both optimized by NSGA-II.
      (binding is now part of the chromosome, one binding per scenario)
    - Communication is currently ignored.
    - Each candidate architecture is evaluated on all scenarios.
    - Overall SADF throughput = minimum scenario throughput.
    - Objectives:
        1. Maximize worst-case SADF throughput
        2. Minimize architecture cost
"""

import random
import numpy as np
import os
import xml.etree.ElementTree as ET
import subprocess
from xml.dom import minidom
from deap import base, creator, tools, algorithms

import matplotlib.pyplot as plt


# ==============================
# Constants
# ==============================

XML_OUTPUT_DIR = "sadf_xml_files"
EVOLUTION_PLOTS_DIR = "sadf_evolution_plot"


def cleanup_old_xml_files():
    """Remove old XML files directory if it exists"""

    if os.path.exists(XML_OUTPUT_DIR):
        import shutil
        shutil.rmtree(XML_OUTPUT_DIR)

    os.makedirs(XML_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EVOLUTION_PLOTS_DIR, exist_ok=True)


# ==============================
# Problem Container
# ==============================

class FlexibleArchProblem:

    def __init__(self):

        self.tasks = ["a", "b", "c", "d"]
        self.n_tasks = len(self.tasks)

        self.pe_types = ["fpga", "gpp", "asic", "dsp"]
        self.n_types = len(self.pe_types)

        self.pe_cost = {
            "fpga": 100,
            "gpp": 50,
            "asic": 150,
            "dsp": 80
        }

        self.exec_time_table = np.array([
            [1, 2, 1, 3],
            [2, 1, 3, 1],
            [1, 1, 2, 2],
            [3, 2, 2, 1],
        ], dtype=float)

        # ==========================================================
        # Each scenario has its own rates_matrix.
        # ==========================================================

        self.scenario_rates = {

            "s1": [
                #   a              b              c              d
                [(0,0,0),   (2,1,0),   (2,1,0),   (0,0,0)],
                [(0,0,0),   (0,0,0),   (2,2,0),   (2,2,0)],
                [(0,0,0),   (0,0,0),   (0,0,0),   (2,2,0)],
                [(1,2,2),   (0,0,0),   (0,0,0),   (0,0,0)]
            ],

            "s2": [
                #   a              b              c              d
                [(0,0,0),   (1,1,0),   (1,1,0),   (0,0,0)],
                [(0,0,0),   (0,0,0),   (1,1,0),   (1,1,0)],
                [(0,0,0),   (0,0,0),   (0,0,0),   (1,1,0)],
                [(1,1,1),   (0,0,0),   (0,0,0),   (0,0,0)]
            ],

            "s3": [
                #   a              b              c              d
                [(0,0,0),   (3,1,0),   (3,1,0),   (0,0,0)],
                [(0,0,0),   (0,0,0),   (2,2,0),   (2,2,0)],
                [(0,0,0),   (0,0,0),   (0,0,0),   (2,2,0)],
                [(1,3,3),   (0,0,0),   (0,0,0),   (0,0,0)]
            ]
        }

        self.max_alloc = 4

        # ==========================================================
        # NEW:
        # Binding is now part of the chromosome instead of a fixed
        # table. We just need a stable ordering of scenarios and the
        # number of binding genes (one gene per task, per scenario).
        # ==========================================================

        self.scenario_names = list(self.scenario_rates.keys())   # ["s1", "s2", "s3"]
        self.n_scenarios = len(self.scenario_names)

        # one binding gene per (scenario, task) pair
        self.n_binding_genes = self.n_scenarios * self.n_tasks

        # total chromosome length = allocation genes + binding genes
        self.chromosome_length = self.n_types + self.n_binding_genes


    def allocation_to_platform(self, alloc_vector):

        """
        Convert an allocation vector (how many PEs of each type) into
        an explicit list of physical PEs.

        Example:
            alloc_vector = [2, 1, 0, 1]   (fpga, gpp, asic, dsp)
            -> platform  = ["fpga", "fpga", "gpp", "dsp"]

        This list's INDEX is the physical tile id used everywhere else
        (in binding_to_mapping, in the generated SDF3 XML, etc).
        """

        platform = []

        for t_index, cnt in enumerate(alloc_vector):

            for _ in range(int(max(1, cnt))):

                platform.append(
                    self.pe_types[t_index]
                )

        return platform


    def binding_to_mapping(self, platform, binding):

        """
        Convert a symbolic binding (task -> PE TYPE, e.g. "a": "fpga")
        into a physical mapping (task -> physical tile index, e.g. "a": 2).

        binding says WHAT KIND of PE a task must run on.
        mapping says WHICH SPECIFIC physical tile it runs on.

        We need this translation because the chromosome/binding only
        knows about PE *types*, not about how many physical tiles of
        each type actually exist in this candidate architecture -- and
        SDF3 needs a concrete tile index for every actor.
        """

        mapping = {}
        type_counter = {}

        for task in self.tasks:

            pe_type = binding[task]

            # how many tasks (before this one) already asked for this type
            idx = type_counter.get(pe_type, 0)
            type_counter[pe_type] = idx + 1

            candidate_pes = [
                pe_index
                for pe_index, p_type in enumerate(platform)
                if p_type == pe_type
            ]

            if not candidate_pes:
                # This platform has zero PEs of the required type ->
                # this candidate architecture cannot run this binding.
                return None

            # Distribute tasks that need the same PE type across all
            # available tiles of that type, round-robin, using an
            # index that is local to this PE type (not the task's
            # global index) so it actually spreads them out.
            mapping[task] = candidate_pes[idx % len(candidate_pes)]

        return mapping


# ==============================
# NEW: Decode binding genes from chromosome
# ==============================

def decode_bindings(individual):
    """
    Chromosome layout:

        [ alloc genes (n_types) ] + [ binding genes (n_scenarios * n_tasks) ]

    The binding genes are small integers in [0, n_types - 1] that are
    decoded into PE type names. Returns:

        { scenario_name: { task: pe_type_string } }
    """

    binding_genes = individual[
        problem.n_types : problem.n_types + problem.n_binding_genes
    ]

    bindings = {}
    gene_idx = 0

    for scenario_name in problem.scenario_names:

        binding = {}

        for task in problem.tasks:

            type_index = int(binding_genes[gene_idx]) % problem.n_types
            binding[task] = problem.pe_types[type_index]
            gene_idx += 1

        bindings[scenario_name] = binding

    return bindings


# ==============================
# SDF Application Classes
# ==============================

class Actor:

    def __init__(
        self,
        name,
        exec_time
    ):

        self.name = name
        self.exec_time = exec_time
        self.in_ports = {}
        self.out_ports = {}


class Channel:

    def __init__(
        self,
        src_actor,
        src_port,
        dst_actor,
        dst_port,
        init_tokens=0
    ):

        self.src_actor = src_actor
        self.src_port = src_port
        self.dst_actor = dst_actor
        self.dst_port = dst_port
        self.init_tokens = init_tokens


class SDFApplication:

    def __init__(
        self,
        name="app"
    ):

        self.name = name
        self.actors = {}
        self.channels = []


# ==============================
# XML + SDF3
# ==============================

def generate_sdf3_xml(
    app,
    platform,
    mapping,
    filename
):

    """Generate SDF3 XML file from application, platform and mapping"""

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )


    sdf3_el = ET.Element(
        "sdf3",
        {
            "xmlns:xsi":
                "http://www.w3.org/2001/XMLSchema-instance",

            "type":
                "sdf",

            "version":
                "1.0",

            "xsi:noNamespaceSchemaLocation":
                "/mnt/d/SDF3/sdf3/sdf/xsd/sdf3-sdf.xsd"
        }
    )


    app_graph = ET.SubElement(
        sdf3_el,
        "applicationGraph",
        name=app.name
    )


    sdf_el = ET.SubElement(
        app_graph,
        "sdf",
        name=app.name,
        type="SDF"
    )


    arch_graph = ET.SubElement(
        sdf3_el,
        "architectureGraph",
        name="arch"
    )


    # ==============================
    # Architecture tiles
    # ==============================

    for i, pe_type in enumerate(platform):

        tile = ET.SubElement(
            arch_graph,
            "tile",
            name=f"t{i}"
        )

        ET.SubElement(
            tile,
            "processor",
            name=f"p{i}",
            type=pe_type
        )

        ET.SubElement(
            tile,
            "memory",
            name=f"m{i}",
            size="1024"
        )

        ET.SubElement(
            tile,
            "networkInterface",
            name=f"ni{i}"
        )


    # ==============================
    # Mapping
    # ==============================

    mapping_el = ET.SubElement(
        sdf3_el,
        "mapping",
        appGraph=app.name,
        archGraph="arch"
    )


    tile_map = {}


    for task, pe_index in mapping.items():

        tile_name = f"t{pe_index}"

        tile_map.setdefault(
            tile_name,
            []
        ).append(task)


    for tile_name, actors in tile_map.items():

        tile_el = ET.SubElement(
            mapping_el,
            "tile",
            name=tile_name
        )

        for actor_name in actors:

            ET.SubElement(
                tile_el,
                "actor",
                name=actor_name
            )


    # ==============================
    # Actors with ports
    # ==============================

    for actor in app.actors.values():

        a_el = ET.SubElement(
            sdf_el,
            "actor",
            name=actor.name,
            type=actor.name
        )


        for port, rate in actor.in_ports.items():

            ET.SubElement(
                a_el,
                "port",
                name=port,
                type="in",
                rate=str(rate)
            )


        for port, rate in actor.out_ports.items():

            ET.SubElement(
                a_el,
                "port",
                name=port,
                type="out",
                rate=str(rate)
            )


    # ==============================
    # Channels
    # ==============================

    for idx, ch in enumerate(
        app.channels,
        start=1
    ):

        ch_name = f"ch{idx}"


        attrs = {
            "name": ch_name,
            "srcActor": ch.src_actor,
            "srcPort": ch.src_port,
            "dstActor": ch.dst_actor,
            "dstPort": ch.dst_port
        }


        if ch.init_tokens > 0:

            attrs["initialTokens"] = str(
                ch.init_tokens
            )


        ET.SubElement(
            sdf_el,
            "channel",
            **attrs
        )


    # ==============================
    # Properties
    # ==============================

    sdf_props = ET.SubElement(
        app_graph,
        "sdfProperties"
    )


    # Actor properties

    for actor in app.actors.values():

        actor_prop = ET.SubElement(
            sdf_props,
            "actorProperties",
            actor=actor.name
        )


        assigned_processor_type = None


        for pe_index, pe_type in enumerate(platform):

            if mapping.get(actor.name) == pe_index:

                assigned_processor_type = pe_type
                break


        if (
            assigned_processor_type
            and
            assigned_processor_type in actor.exec_time
        ):

            proc = ET.SubElement(
                actor_prop,
                "processor",
                type=assigned_processor_type,
                default="true"
            )


            ET.SubElement(
                proc,
                "executionTime",
                time=str(
                    actor.exec_time[
                        assigned_processor_type
                    ]
                )
            )


        else:

            first_type = (
                list(
                    actor.exec_time.keys()
                )[0]
                if actor.exec_time
                else "gpp"
            )


            proc = ET.SubElement(
                actor_prop,
                "processor",
                type=first_type,
                default="true"
            )


            ET.SubElement(
                proc,
                "executionTime",
                time=str(
                    actor.exec_time.get(
                        first_type,
                        1
                    )
                )
            )


    # Channel properties

    for idx, ch in enumerate(
        app.channels,
        start=1
    ):

        ch_name = f"ch{idx}"

        ET.SubElement(
            sdf_props,
            "channelProperties",
            channel=ch_name
        )


    ET.SubElement(
        sdf_props,
        "graphProperties"
    )


    # ==============================
    # Pretty print
    # ==============================

    rough_string = ET.tostring(
        sdf3_el,
        "utf-8"
    )


    reparsed = minidom.parseString(
        rough_string
    )


    pretty_xml = reparsed.toprettyxml(
        indent="  "
    )


    pretty_xml = "\n".join(
        [
            line
            for line in pretty_xml.splitlines()
            if line.strip()
        ]
    )


    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(pretty_xml)


def run_sdf3(xml_file):

    """Run SDF3 throughput analysis on XML file"""

    cmd = [

        "/mnt/d/SDF3/sdf3/build/release/Linux/bin/"
        "sdf3analysis-sdf",

        "--graph",
        xml_file,

        "--algo",
        "throughput"
    ]


    env = os.environ.copy()


    env["LD_LIBRARY_PATH"] = (
        "/mnt/d/SDF3/sdf3/build/release/Linux/lib:"
        + env.get(
            "LD_LIBRARY_PATH",
            ""
        )
    )


    try:

        output = subprocess.check_output(
            cmd,
            env=env,
            stderr=subprocess.STDOUT
        ).decode()


        return parse_throughput(
            output
        )


    except subprocess.CalledProcessError as e:

        print(
            "SDF3 error: "
            +
            (
                e.output.decode()
                if e.output
                else "Unknown error"
            )
        )

        return 0.0


    except Exception as e:

        print(
            f"Unexpected error: {e}"
        )

        return 0.0


def parse_throughput(output):

    """Parse throughput value from SDF3 output"""

    for line in output.splitlines():

        line = line.strip()


        if line.startswith("thr("):

            val = line.split("=")[-1].strip()


            if val == "inf":

                return 1e6


            try:

                return float(val)


            except ValueError:

                continue


    return 0.0


# ==============================
# DEAP Setup
# ==============================

problem = FlexibleArchProblem()


creator.create(
    "FitnessMulti",
    base.Fitness,
    weights=(1.0, -1.0)
)


creator.create(
    "Individual",
    list,
    fitness=creator.FitnessMulti
)


toolbox = base.Toolbox()


# ==========================================================
# Chromosome layout:
#
#   [ alloc genes (n_types) ] + [ binding genes (n_scenarios * n_tasks) ]
#
# Example (n_types=4, n_scenarios=3, n_tasks=4):
#
#   [2, 1, 1, 2,           <- allocation: fpga, gpp, asic, dsp counts
#    0, 1, 0, 3,           <- s1 binding: a->fpga, b->gpp, c->fpga, d->dsp
#    1, 0, 3, 1,           <- s2 binding
#    2, 3, 1, 0]           <- s3 binding
#
# Both allocation AND binding are now optimized by NSGA-II.
# ==========================================================

def init_individual():
    """Initialize individual: allocation AND binding are both optimized."""

    alloc = np.random.randint(
        1,
        problem.max_alloc + 1,
        size=problem.n_types
    )

    # NEW: random binding genes (one per scenario per task)
    binding_genes = np.random.randint(
        0,
        problem.n_types,
        size=problem.n_binding_genes
    )

    genome = list(alloc) + list(binding_genes)

    return creator.Individual(genome)


toolbox.register(
    "individual",
    init_individual
)


toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)


# ==============================
# Custom Crossover
# ==============================

def custom_crossover(
    ind1,
    ind2
):

    """
    Crossover over the FULL chromosome (allocation genes + binding genes).
    A single crossover point is picked anywhere in the whole genome, so
    offspring can mix allocation and binding material freely.
    """

    total_len = problem.chromosome_length

    point = random.randint(
        1,
        total_len - 1
    )

    ind1[:point], ind2[:point] = (
        ind2[:point],
        ind1[:point]
    )

    # Repair only the allocation part.
    # Binding genes are always valid integers in [0, n_types - 1]
    # after crossover (they were valid before, and swapping doesn't
    # break that), so they need no repair.

    for ind in (
        ind1,
        ind2
    ):

        for i in range(problem.n_types):

            ind[i] = max(
                1,
                min(
                    problem.max_alloc,
                    int(ind[i])
                )
            )

    return ind1, ind2


# ==============================
# Custom Mutation
# ==============================

def custom_mutation(ind):

    """
    Mutation for allocation genes AND binding genes.
    """

    # Mutate allocation

    for i in range(problem.n_types):

        if random.random() < 0.3:

            delta = random.choice(
                [-1, 1]
            )

            ind[i] = max(
                1,
                min(
                    problem.max_alloc,
                    ind[i] + delta
                )
            )

    # NEW: Mutate binding genes -- each gene has a 30% chance of being
    # reassigned to a random PE type.

    for i in range(
        problem.n_types,
        problem.n_types + problem.n_binding_genes
    ):

        if random.random() < 0.3:

            ind[i] = random.randint(
                0,
                problem.n_types - 1
            )

    return ind,


toolbox.register(
    "mate",
    custom_crossover
)


toolbox.register(
    "mutate",
    custom_mutation
)


toolbox.register(
    "select",
    tools.selNSGA2
)


# ==============================
# Scenario Application Builder
# ==============================

def build_scenario_application(
    scenario_name
):

    """
    Build an SDF application (actors + channels) from one scenario's
    rates_matrix. This is completely independent of platform/binding --
    it only describes the algorithm graph.
    """

    rates_matrix = problem.scenario_rates[
        scenario_name
    ]


    app = SDFApplication(
        scenario_name
    )


    # Create actors

    for i, task in enumerate(
        problem.tasks
    ):

        exec_times = {}


        for j, pe_type in enumerate(
            problem.pe_types
        ):

            exec_times[pe_type] = (
                problem.exec_time_table[i][j]
            )


        actor = Actor(
            task,
            exec_times
        )


        app.actors[task] = actor


    # Create channels

    for i in range(
        problem.n_tasks
    ):

        for j in range(
            problem.n_tasks
        ):

            (
                prod_rate,
                cons_rate,
                init_tokens
            ) = rates_matrix[i][j]


            if (
                prod_rate > 0
                and
                cons_rate > 0
            ):

                src_actor = problem.tasks[i]
                dst_actor = problem.tasks[j]


                src_port = (
                    f"out_{i}_{j}"
                )

                dst_port = (
                    f"in_{i}_{j}"
                )


                channel = Channel(
                    src_actor,
                    src_port,
                    dst_actor,
                    dst_port,
                    init_tokens
                )


                app.channels.append(
                    channel
                )


                app.actors[
                    src_actor
                ].out_ports[
                    src_port
                ] = prod_rate


                app.actors[
                    dst_actor
                ].in_ports[
                    dst_port
                ] = cons_rate


    return app


# ==============================
# Evaluation
# ==============================

def evaluate(
    individual,
    generation,
    candidate_index
):

    """
    Evaluate one architecture (allocation + binding, both taken from
    the chromosome) on all SADF scenarios.
    """

    alloc = individual[:problem.n_types]

    # Build architecture (list of physical PEs)

    platform = problem.allocation_to_platform(alloc)

    # NEW: decode all scenario bindings from this individual's chromosome
    bindings = decode_bindings(individual)

    generation_dir = os.path.join(
        XML_OUTPUT_DIR,
        f"generation_{generation:03d}"
    )


    os.makedirs(
        generation_dir,
        exist_ok=True
    )


    # Store throughput of every scenario

    scenario_throughputs = {}


    # Evaluate every scenario independently.

    for scenario_name in problem.scenario_names:


        # CHANGED: binding now comes from the chromosome (evolved),
        # not from a fixed table.

        binding = bindings[scenario_name]


        # Convert PE-type binding into physical mapping

        mapping = problem.binding_to_mapping(platform, binding)


        # If required PE type does not exist,
        # candidate is invalid.

        if mapping is None:

            return (
                0.0,
                1e9
            )


        # Build scenario-specific SDF

        app = build_scenario_application(
            scenario_name
        )


        xml_filename = (
            f"candidate_{candidate_index:03d}_"
            f"{scenario_name}.xml"
        )


        full_xml_path = os.path.join(
            generation_dir,
            xml_filename
        )


        generate_sdf3_xml(
            app,
            platform,
            mapping,
            full_xml_path
        )


        # Run SDF3

        throughput = run_sdf3(
            full_xml_path
        )


        # Save scenario throughput

        scenario_throughputs[
            scenario_name
        ] = throughput


    # Overall SADF throughput is defined as
    # the worst-case scenario throughput.

    overall_throughput = min(
        scenario_throughputs.values()
    )


    # Architecture cost

    cost = sum(
        alloc[i]
        *
        problem.pe_cost[
            problem.pe_types[i]
        ]

        for i in range(
            problem.n_types
        )
    )


    return (
        overall_throughput,
        cost
    )


# ==============================
# Plot Utilities
# ==============================

def save_evolution_plot(
    population,
    generation
):

    """Save plot for each generation"""


    os.makedirs(
        EVOLUTION_PLOTS_DIR,
        exist_ok=True
    )


    valid_individuals = [
        ind
        for ind in population
        if ind.fitness.valid
    ]


    if not valid_individuals:

        return


    throughputs = [
        ind.fitness.values[0]
        for ind in valid_individuals
    ]


    costs = [
        ind.fitness.values[1]
        for ind in valid_individuals
    ]


    # Pareto front

    non_dominated = tools.sortNondominated(
        valid_individuals,
        k=len(valid_individuals),
        first_front_only=True
    )[0]


    nd_throughputs = [
        ind.fitness.values[0]
        for ind in non_dominated
    ]


    nd_costs = [
        ind.fitness.values[1]
        for ind in non_dominated
    ]


    sorted_pairs = sorted(
        zip(
            nd_throughputs,
            nd_costs
        )
    )


    if sorted_pairs:

        (
            nd_throughputs_sorted,
            nd_costs_sorted
        ) = zip(*sorted_pairs)

    else:

        nd_throughputs_sorted = []
        nd_costs_sorted = []


    # Plot

    plt.figure(
        figsize=(10, 6)
    )


    plt.scatter(
        throughputs,
        costs,
        alpha=0.6,
        c="blue",
        s=50,
        label=(
            f"Population "
            f"(n={len(valid_individuals)})"
        ),
        edgecolors="black",
        linewidth=0.5
    )


    if nd_throughputs_sorted:

        plt.scatter(
            nd_throughputs_sorted,
            nd_costs_sorted,
            c="red",
            s=150,
            marker="*",
            label=(
                f"Pareto Front "
                f"(n={len(nd_throughputs_sorted)})"
            ),
            edgecolors="darkred",
            linewidth=1.5,
            zorder=5
        )


        plt.plot(
            nd_throughputs_sorted,
            nd_costs_sorted,
            "r--",
            alpha=0.5,
            linewidth=1
        )


    plt.xlabel(
        "Overall SADF Throughput",
        fontsize=12,
        fontweight="bold"
    )


    plt.ylabel(
        "Cost",
        fontsize=12,
        fontweight="bold"
    )


    plt.title(
        f"Generation {generation} - "
        f"Population vs Pareto Front",
        fontsize=14,
        fontweight="bold"
    )


    plt.legend(
        loc="best",
        framealpha=0.9,
        fontsize=10
    )


    plt.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )


    if throughputs:

        info_text = (
            f"Best Throughput: "
            f"{max(throughputs):.4f}\n"
            f"Best Cost: "
            f"{min(costs):.2f}"
        )


        plt.text(
            0.02,
            0.98,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="wheat",
                alpha=0.5
            )
        )


    plt.tight_layout()


    filename = (
        f"{EVOLUTION_PLOTS_DIR}/"
        f"generation_{generation:03d}.png"
    )


    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


# ==============================
# Main Evolution Loop
# ==============================

def main():

    cleanup_old_xml_files()


    POP_SIZE = 30

    NGEN = 20

    MU = POP_SIZE

    CXPB = 0.9

    MUTPB = 0.5


    pop = toolbox.population(
        n=POP_SIZE
    )


    # Evaluate initial population.
    #
    # Candidate index is explicitly passed so XML files can
    # be stored as candidate_000, candidate_001, ...

    for candidate_index, ind in enumerate(pop):

        fit = evaluate(
            ind,
            generation=0,
            candidate_index=candidate_index
        )

        ind.fitness.values = fit


    # Save generation 0 plot

    save_evolution_plot(
        pop,
        0
    )


    # ==============================
    # Evolution loop
    # ==============================

    for gen in range(
        1,
        NGEN + 1
    ):


        # Generate offspring

        offspring = algorithms.varAnd(
            pop,
            toolbox,
            CXPB,
            MUTPB
        )


        for candidate_index, ind in enumerate(
            offspring
        ):

            fit = evaluate(
                ind,
                generation=gen,
                candidate_index=candidate_index
            )


            ind.fitness.values = fit


        # Select next generation

        pop = toolbox.select(
            pop + offspring,
            MU
        )


        # Save plot

        save_evolution_plot(
            pop,
            gen
        )


        # Progress information

        best_throughput = max(
            ind.fitness.values[0]
            for ind in pop
            if ind.fitness.valid
        )


        best_cost = min(
            ind.fitness.values[1]
            for ind in pop
            if ind.fitness.valid
        )


        print(
            f"Generation {gen}: "
            f"Best Throughput="
            f"{best_throughput:.4f} | "
            f"Best Cost="
            f"{best_cost:.2f}"
        )


    # ==============================
    # Final Pareto Front
    # ==============================

    front = tools.sortNondominated(
        pop,
        k=len(pop),
        first_front_only=True
    )[0]


    print(
        "\n"
        + "=" * 70
    )


    print(
        "Final SADF Pareto Front:"
    )


    print(
        "=" * 70
    )


    # Print unique Pareto solutions only.
    #
    # This prevents printing many identical copies of the same
    # architecture+binding when several individuals converge to it.
    #
    # NEW: the "key" now includes the binding genes too (not just
    # allocation), since two individuals with the same allocation but
    # different bindings are different solutions.

    unique_solutions = {}


    for ind in front:

        genome_key = tuple(ind[:problem.chromosome_length])


        fitness = (
            ind.fitness.values[0],
            ind.fitness.values[1]
        )


        unique_solutions[
            genome_key
        ] = fitness


    sorted_solutions = sorted(
        unique_solutions.items(),
        key=lambda x: x[1][0],
        reverse=True
    )


    for i, (
        genome_key,
        fitness
    ) in enumerate(
        sorted_solutions
    ):

        throughput = fitness[0]
        cost = fitness[1]

        architecture = list(genome_key[:problem.n_types])

        # NEW: decode the bindings for display purposes
        fake_ind = list(genome_key)
        bindings = decode_bindings(fake_ind)


        print(
            f"{i + 1}. "
            f"Architecture={architecture} | "
            f"Throughput={throughput:.6f} | "
            f"Cost={cost:.2f}"
        )

        for scenario_name in problem.scenario_names:

            print(
                f"     {scenario_name} binding: "
                f"{bindings[scenario_name]}"
            )


if __name__ == "__main__":

    main()