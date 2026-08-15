"""
Multi-Objective Hardware/Software Co-Design for SADF Applications using NSGA-II
================================================================================

This version extends the original SDF implementation to a simple SADF model.

Main idea:
    - The chromosome contains ONLY the allocation vector.
    - Architecture is optimized by NSGA-II.
    - Each scenario has:
        1. Its own SDF rates_matrix
        2. Its own fixed binding defined by PE TYPE
    - For each candidate architecture, every scenario is evaluated separately
      using SDF3.
    - The overall SADF throughput is currently defined as the worst-case
      scenario throughput:
            overall_throughput = min(scenario_throughputs)
    - Fitness:
            maximize overall throughput
            minimize total architecture cost

Important:
    This is a simple SADF prototype. It evaluates each scenario independently
    and does not yet model scenario transitions, scenario graph behavior,
    communication costs, or scenario probabilities.
"""

import random
import numpy as np
import os
import xml.etree.ElementTree as ET
import subprocess
from xml.dom import minidom
from deap import base, creator, tools, algorithms

import matplotlib.pyplot as plt

from scoop import futures  # for parallel processing


# ==============================
# Constants
# ==============================

XML_OUTPUT_DIR = "sdf_xml_files"
EVOLUTION_PLOTS_DIR = "sdf_evolution_plot"


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
            [1, 2, 1, 3],   # actor a
            [2, 1, 3, 1],   # actor b
            [1, 1, 2, 2],   # actor c
            [3, 2, 2, 1],   # actor d
        ], dtype=float)


        # ==========================================================
        # CHANGE: Instead of one rates_matrix, define one per scenario
        # ==========================================================

        self.scenario_rates = {

            # ------------------------------------------------------
            # Scenario 1
            # Repetition vector approximately [1,2,2,2]
            # ------------------------------------------------------
            "s1": [
                #   a              b              c              d
                [(0,0,0),   (2,1,0),   (2,1,0),   (0,0,0)],
                [(0,0,0),   (0,0,0),   (2,2,0),   (2,2,0)],
                [(0,0,0),   (0,0,0),   (0,0,0),   (2,2,0)],
                [(1,2,2),   (0,0,0),   (0,0,0),   (0,0,0)]
            ],

            # ------------------------------------------------------
            # Scenario 2
            # Simple SDF with all production/consumption rates = 1
            # Repetition vector = [1,1,1,1]
            # ------------------------------------------------------
            "s2": [
                #   a              b              c              d
                [(0,0,0),   (1,1,0),   (1,1,0),   (0,0,0)],
                [(0,0,0),   (0,0,0),   (1,1,0),   (1,1,0)],
                [(0,0,0),   (0,0,0),   (0,0,0),   (1,1,0)],
                [(1,1,1),   (0,0,0),   (0,0,0),   (0,0,0)]
            ],

            # ------------------------------------------------------
            # Scenario 3
            #
            # a -> b : 3 / 1
            # b -> c : 2 / 2
            # c -> d : 2 / 2
            # d -> a : 1 / 3
            #
            # One valid repetition vector is [1,3,3,3]
            # ------------------------------------------------------
            "s3": [
                #   a              b              c              d
                [(0,0,0),   (3,1,0),   (3,1,0),   (0,0,0)],
                [(0,0,0),   (0,0,0),   (2,2,0),   (2,2,0)],
                [(0,0,0),   (0,0,0),   (0,0,0),   (2,2,0)],
                [(1,3,3),   (0,0,0),   (0,0,0),   (0,0,0)]
            ]
        }


        # ==========================================================
        # CHANGE: Fixed binding for each scenario.
        #
        # Binding is defined by PE TYPE, NOT physical processor ID.
        #
        # Example:
        #     "a": "fpga"
        #
        # means actor a is always bound to an FPGA.
        # ==========================================================

        self.scenario_bindings = {

            "s1": {
                "a": "fpga",
                "b": "gpp",
                "c": "fpga",
                "d": "dsp"
            },

            "s2": {
                "a": "gpp",
                "b": "fpga",
                "c": "dsp",
                "d": "gpp"
            },

            "s3": {
                "a": "asic",
                "b": "dsp",
                "c": "gpp",
                "d": "fpga"
            }
        }


        self.max_alloc = 4


    def allocation_to_platform(self, alloc_vector):
        """Convert allocation vector to platform list of PE types"""

        platform = []

        for t_index, cnt in enumerate(alloc_vector):

            for _ in range(int(max(1, cnt))):
                platform.append(self.pe_types[t_index])

        return platform


    # ==========================================================
    # CHANGE:
    # New binding function.
    #
    # Binding is specified using PE TYPES.
    #
    # For example:
    #     binding["a"] = "fpga"
    #
    # If architecture contains multiple FPGAs, the first FPGA
    # is selected.
    # ==========================================================

    def binding_type_to_mapping(self, platform, binding):

        mapping = {}

        for task in self.tasks:

            required_pe_type = binding[task]

            # Find the first processor having this PE type
            selected_processor = None

            for pe_index, pe_type in enumerate(platform):

                if pe_type == required_pe_type:
                    selected_processor = pe_index
                    break

            # --------------------------------------------------
            # If the architecture does not contain the required
            # PE type, this scenario/architecture combination
            # is invalid.
            # --------------------------------------------------
            if selected_processor is None:
                return None

            mapping[task] = selected_processor

        return mapping


# ==============================
# SDF Application Classes
# ==============================

class Actor:

    def __init__(self, name, exec_time):

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

    def __init__(self, name="app"):

        self.name = name
        self.actors = {}
        self.channels = []


# ==============================
# XML + SDF3
# ==============================

def generate_sdf3_xml(app, platform, mapping, filename):

    """Generate SDF3 XML file from application, platform and mapping"""

    os.makedirs(XML_OUTPUT_DIR, exist_ok=True)

    if not os.path.dirname(filename):
        filename = os.path.join(XML_OUTPUT_DIR, filename)


    sdf3_el = ET.Element(
        "sdf3",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "type": "sdf",
            "version": "1.0",
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


    # Architecture tiles

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


    # Mapping

    mapping_el = ET.SubElement(
        sdf3_el,
        "mapping",
        appGraph="app",
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


    # Actors with ports

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


    # Channels

    for idx, ch in enumerate(app.channels, start=1):

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


    # Properties

    sdf_props = ET.SubElement(
        app_graph,
        "sdfProperties"
    )


    # Actor properties with execution times

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
            and assigned_processor_type in actor.exec_time
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
                list(actor.exec_time.keys())[0]
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

    for idx, ch in enumerate(app.channels, start=1):

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


    # Pretty print

    rough_string = ET.tostring(
        sdf3_el,
        'utf-8'
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
        "/mnt/d/SDF3/sdf3/build/release/Linux/bin/sdf3analysis-sdf",
        "--graph",
        xml_file,
        "--algo",
        "throughput"
    ]

    env = os.environ.copy()

    env["LD_LIBRARY_PATH"] = (
        "/mnt/d/SDF3/sdf3/build/release/Linux/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )


    try:

        output = subprocess.check_output(
            cmd,
            env=env,
            stderr=subprocess.STDOUT
        ).decode()

        return parse_throughput(output)


    except subprocess.CalledProcessError as e:

        print(
            f"SDF3 error: "
            f"{e.output.decode() if e.output else 'Unknown error'}"
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
# CHANGE:
# Chromosome now contains ONLY the allocation vector.
#
# Original:
#     [allocation | binding]
#
# New:
#     [allocation]
#
# Binding is fixed for each scenario.
# ==========================================================

def init_individual():

    """Initialize a random architecture individual"""

    alloc = np.random.randint(
        1,
        problem.max_alloc + 1,
        size=problem.n_types
    )

    return creator.Individual(
        list(alloc)
    )


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

def custom_crossover(ind1, ind2):

    """
    Crossover for allocation-only individuals.
    """

    n_types = problem.n_types


    # Crossover allocation part

    point = random.randint(
        1,
        n_types - 1
    )

    ind1[:point], ind2[:point] = (
        ind2[:point],
        ind1[:point]
    )


    # ==========================================================
    # CHANGE:
    # Binding crossover was removed because binding is fixed
    # per scenario and is no longer part of chromosome.
    # ==========================================================


    # ==========================================================
    # CHANGE:
    # Allocation repair.
    # ==========================================================

    for ind in (ind1, ind2):

        for i in range(n_types):

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
    Mutation for allocation-only individuals.
    """

    n_types = problem.n_types


    # Mutate allocation

    for i in range(n_types):

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


    # ==========================================================
    # CHANGE:
    # Binding mutation was completely removed.
    # Binding is fixed for each scenario.
    # ==========================================================


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
# Evaluation
# ==============================

# ==========================================================
# CHANGE:
# New helper function.
#
# It builds ONE SDF application for ONE scenario.
# ==========================================================

def build_scenario_application(scenario_name):

    rates_matrix = problem.scenario_rates[
        scenario_name
    ]

    app = SDFApplication(
        scenario_name
    )


    # Create actors

    for i, task in enumerate(problem.tasks):

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


    # Create channels from scenario-specific rates_matrix

    for i in range(problem.n_tasks):

        for j in range(problem.n_tasks):

            prod_rate, cons_rate, init_tokens = (
                rates_matrix[i][j]
            )

            if (
                prod_rate > 0
                and cons_rate > 0
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


# ==========================================================
# CHANGE:
# evaluate() now evaluates the SAME candidate architecture
# for ALL SADF scenarios.
# ==========================================================

def evaluate(individual):

    """
    Evaluate one SADF candidate.

    Individual:
        allocation vector

    For each scenario:
        1. Use fixed scenario binding
        2. Convert PE-type binding to physical processor mapping
        3. Build scenario-specific SDF
        4. Generate XML
        5. Run SDF3
        6. Get scenario throughput

    Overall throughput:
        min(throughput of all scenarios)

    Fitness:
        (overall throughput, cost)
    """


    # ==========================================================
    # CHANGE:
    # Chromosome contains only allocation.
    # ==========================================================

    alloc = individual[:problem.n_types]


    # Build architecture

    platform = problem.allocation_to_platform(
        alloc
    )


    # ==========================================================
    # CHANGE:
    Store throughput of every scenario.
    # This is useful later for SADF/safety analysis.
    # ==========================================================

    scenario_throughputs = {}


    # ==========================================================
    # CHANGE:
    Evaluate every scenario separately.
    # ==========================================================

    for scenario_name in problem.scenario_rates.keys():

        # ----------------------------------------------
        # Fixed binding of this scenario
        # ----------------------------------------------

        binding = problem.scenario_bindings[
            scenario_name
        ]


        # ----------------------------------------------
        # Convert type-based binding into physical
        # processor mapping for this architecture.
        # ----------------------------------------------

        mapping = problem.binding_type_to_mapping(
            platform,
            binding
        )


        # ==================================================
        # CHANGE:
        # If architecture does not contain one of the
        # required PE types, this candidate is invalid.
        #
        # Since every scenario in this example requires
        # FPGA, GPP, ASIC and DSP, all allocation genes
        # are >= 1, so normally this will not happen.
        # ==================================================

        if mapping is None:

            return 0.0, 1e9


        # ----------------------------------------------
        # Build SDF for this scenario
        # ----------------------------------------------

        app = build_scenario_application(
            scenario_name
        )


        # ----------------------------------------------
        # Generate XML
        # ----------------------------------------------

        xml_filename = (
            f"{scenario_name}_"
            f"{random.randint(1, 1000000)}.xml"
        )

        full_xml_path = os.path.join(
            XML_OUTPUT_DIR,
            xml_filename
        )


        generate_sdf3_xml(
            app,
            platform,
            mapping,
            full_xml_path
        )


        # ----------------------------------------------
        # Run SDF3
        # ----------------------------------------------

        throughput = run_sdf3(
            full_xml_path
        )


        # Store scenario result

        scenario_throughputs[
            scenario_name
        ] = throughput


        # ----------------------------------------------
        # Clean temporary XML
        # ----------------------------------------------

        try:

            os.remove(
                full_xml_path
            )

        except:

            pass


    # ==========================================================
    # CHANGE:
    # Combine scenario throughputs.
    #
    # Simple prototype definition:
    # SADF throughput = worst-case scenario throughput.
    # ==========================================================

    overall_throughput = min(
        scenario_throughputs.values()
    )


    # ==========================================================
    # Cost is still calculated exactly as in the original code.
    # ==========================================================

    cost = sum(
        alloc[i]
        * problem.pe_cost[
            problem.pe_types[i]
        ]
        for i in range(
            problem.n_types
        )
    )


    # ==========================================================
    # CHANGE:
    # Optional printing of scenario results.
    # You can remove this later because it may produce
    # a lot of output when using SCOOP.
    # ==========================================================

    # print(
    #     f"Architecture={alloc}, "
    #     f"ScenarioThroughputs={scenario_throughputs}, "
    #     f"Overall={overall_throughput}, "
    #     f"Cost={cost}"
    # )


    return (
        overall_throughput,
        cost
    )


toolbox.register(
    "evaluate",
    evaluate
)


# Register SCOOP's parallel map function

toolbox.register(
    "map",
    futures.map
)


# ==============================
# Plot Utilities
# ==============================

def save_evolution_plot(
    population,
    generation
):

    """Save plot for each generation with Pareto front"""


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


    # Find Pareto front

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


    # Sort Pareto front for better visualization

    sorted_pairs = sorted(
        zip(
            nd_throughputs,
            nd_costs
        )
    )


    nd_throughputs_sorted, nd_costs_sorted = (
        zip(*sorted_pairs)
        if sorted_pairs
        else ([], [])
    )


    # Create plot

    plt.figure(
        figsize=(10, 6)
    )


    # Draw population

    plt.scatter(
        throughputs,
        costs,
        alpha=0.6,
        c='blue',
        s=50,
        label=(
            f'Population '
            f'(n={len(valid_individuals)})'
        ),
        edgecolors='black',
        linewidth=0.5
    )


    # Draw Pareto front

    if nd_throughputs_sorted:

        plt.scatter(
            nd_throughputs_sorted,
            nd_costs_sorted,
            c='red',
            s=150,
            marker='*',
            label=(
                f'Pareto Front '
                f'(n={len(nd_throughputs_sorted)})'
            ),
            edgecolors='darkred',
            linewidth=1.5,
            zorder=5
        )


        # Connection line for Pareto front

        plt.plot(
            nd_throughputs_sorted,
            nd_costs_sorted,
            'r--',
            alpha=0.5,
            linewidth=1
        )


    plt.xlabel(
        "Overall SADF Throughput",
        fontsize=12,
        fontweight='bold'
    )

    plt.ylabel(
        "Cost",
        fontsize=12,
        fontweight='bold'
    )

    plt.title(
        f"Generation {generation} - "
        f"Population vs Pareto Front",
        fontsize=14,
        fontweight='bold'
    )

    plt.legend(
        loc='best',
        framealpha=0.9,
        fontsize=10
    )

    plt.grid(
        True,
        alpha=0.3,
        linestyle='--'
    )


    # Add info text

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
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='wheat',
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
        bbox_inches='tight'
    )

    plt.close()


# ==============================
# Main Evolution Loop
# ==============================

def main():

    cleanup_old_xml_files()


    pop = toolbox.population(
        n=30
    )

    NGEN = 20

    MU = 100

    CXPB = 0.9

    MUTPB = 0.5


    # Initial evaluation

    fitnesses = list(
        toolbox.map(
            toolbox.evaluate,
            pop
        )
    )


    for ind, fit in zip(
        pop,
        fitnesses
    ):

        ind.fitness.values = fit


    # Save initial generation plot

    save_evolution_plot(
        pop,
        0
    )


    # Evolution loop

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


        # Evaluate offspring

        fitnesses = list(
            toolbox.map(
                toolbox.evaluate,
                offspring
            )
        )


        for ind, fit in zip(
            offspring,
            fitnesses
        ):

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


        # Print progress

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


    # ==========================================================
    # CHANGE:
    # Final Pareto front now represents architectures.
    # Each architecture was evaluated over ALL scenarios.
    # ==========================================================

    front = tools.sortNondominated(
        pop,
        k=len(pop),
        first_front_only=True
    )[0]


    print(
        "\n" + "=" * 70
    )

    print(
        "Final SADF Pareto Front:"
    )

    print(
        "=" * 70
    )


    for i, ind in enumerate(
        sorted(
            front,
            key=lambda x:
                x.fitness.values[0],
            reverse=True
        )
    ):

        print(
            f"{i+1}. "
            f"Architecture={list(ind)} | "
            f"Throughput="
            f"{ind.fitness.values[0]:.6f} | "
            f"Cost="
            f"{ind.fitness.values[1]:.2f}"
        )


if __name__ == "__main__":

    main()