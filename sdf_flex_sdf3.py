# model sdf
# script for produce xml
# use xml for analyze
# doesn't have scheduler

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import os
import xml.etree.ElementTree as ET
import subprocess
from xml.dom import minidom


###############################################
############# Problem definition ##############
###############################################

class FlexibleArchProblem(ElementwiseProblem):
    def __init__(self,
                 tasks=None,
                 weights=None,
                 pe_types=None,
                 pe_cost=None,
                 exec_time_table=None,
                 tokens_matrix=None,
                 max_alloc_per_type=4,
                 elementwise=True,
                 **kwargs):

        self.tasks = tasks if tasks is not None else ["a", "b", "c", "d"]
        self.n_tasks = len(self.tasks)

        self.weights = weights if weights is not None else {
            ("a", "b"): 0,
            ("a", "c"): 0,
            ("b", "c"): 0,
            ("b", "d"): 0,
            ("c", "d"): 0,
            ("d", "a"): 0
        }

        self.pe_types = pe_types if pe_types is not None else ["fpga", "gpp", "asic", "dsp"]
        self.n_types = len(self.pe_types)
        self.pe_cost = pe_cost if pe_cost is not None else {"fpga": 100, "gpp": 50, "asic": 150, "dsp": 80}

        if exec_time_table is not None:
            self.exec_time_table = exec_time_table
        else:
            self.exec_time_table = np.array([
                [1, 2, 1, 3],
                [2, 1, 3, 1],
                [1, 1, 2, 2],
                [3, 2, 2, 1],
            ], dtype=float)

        self.max_alloc = int(max_alloc_per_type)

        self.tokens_matrix = [
            [(False, 0), (True, 0), (True, 0), (False, 0)],
            [(False, 0), (False, 0), (True, 0), (True, 0)],
            [(False, 0), (False, 0), (False, 0), (True, 0)],
            [(True, 1), (False, 0), (False, 0), (False, 0)]
        ]

        n_var = self.n_types + self.n_tasks

        xl = np.concatenate([
            np.ones(self.n_types),  # allocation >= 1
            np.zeros(self.n_tasks)
        ])

        xu = np.concatenate([
            np.full(self.n_types, self.max_alloc),  # alloc bounds
            np.full(self.n_tasks, 20)  # binding bounds
        ])

        super().__init__(
            elementwise=elementwise,
            n_var=n_var,
            n_obj=2,
            xl=xl,
            xu=xu
        )

    def allocation_to_platform(self, alloc_vector):
        platform = []
        for t_index, cnt in enumerate(alloc_vector):
            if cnt is None:
                continue
            for _ in range(int(max(0, cnt))):
                platform.append(self.pe_types[t_index])
        return platform

        # in => alloc_vector = [A, B, C, ...] & out => platform = ["cpu", "cpu", "cpu", "dsp", "asic", ...]  // based on pe types index

    def binding_to_mapping(self, platform, binding):
        mapping = {}
        P = len(platform)
        if P == 0:
            for i, task in enumerate(self.tasks):
                mapping[task] = 0
            return mapping

        for i, task in enumerate(self.tasks):
            raw = int(binding[i])
            mapping[task] = raw % P
        return mapping

        # platform = ["cpu", "cpu", "cpu", "dsp", "asic", "asic"]
        # binding  = [5, 8, 11, 0]   # raw genes
        # tasks    = ["a", "b", "c", "d"]
        # P = 6
        # mapping[task] = binding[i] % P
        # { "a": 5, "b": 2, "c": 5, "d": 0 }

    class Actor:
        def __init__(self, name, exec_time):
            self.name = name
            self.exec_time = exec_time
            self.in_ports = {}
            self.out_ports = {}

    class Channel:
        def __init__(self, src_actor, src_port, dst_actor, dst_port, init_tokens=0):
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

    def _evaluate(self, x, out, *args, **kwargs):

        alloc = np.array(x[:self.n_types], dtype=int)
        binding = np.array(x[self.n_types:], dtype=int)

        platform = self.allocation_to_platform(alloc)
        mapping = self.binding_to_mapping(platform, binding)

        if len(platform) == 0:
            out["F"] = np.array([1e6, 1e6])
            return

        app = self.SDFApplication("app")

        for t_index, task in enumerate(self.tasks):
            actor = self.Actor(task, {})
            actor.in_ports = {}
            actor.out_ports = {}

            assigned_pe_index = mapping[task]
            assigned_pe_type = platform[assigned_pe_index]

            pe_type_index = self.pe_types.index(assigned_pe_type)
            exec_time = self.exec_time_table[t_index][pe_type_index]

            actor.exec_time = {assigned_pe_type: exec_time}

            app.actors[task] = actor

        for i, src in enumerate(self.tasks):
            for j, dst in enumerate(self.tasks):
                has_edge, tokens = self.tokens_matrix[i][j]
                if has_edge:
                    out_port_name = f"out_{i}_{j}"
                    in_port_name = f"in_{i}_{j}"

                    ch = self.Channel(
                        src_actor=src,
                        src_port=out_port_name,
                        dst_actor=dst,
                        dst_port=in_port_name,
                        init_tokens=tokens
                    )

                    app.channels.append(ch)

                    app.actors[src].out_ports[out_port_name] = 1
                    app.actors[dst].in_ports[in_port_name] = 1

        cost = 0
        pes = []
        for p in mapping.values():
            pes.append(p)

        # del_dup = list(dict.fromkeys(pes))
        # for p in del_dup:
        #     if p < 0 or p >= len(platform):
        #         continue
        #     ptype = platform[p]
        #     cost += self.pe_cost.get(ptype, 0)

        cost = 0
        for t_index, cnt in enumerate(alloc):
            ptype = self.pe_types[t_index]
            cost += cnt * self.pe_cost.get(ptype, 0)

        xml_file = f"tmp_{np.random.randint(1e9)}.xml"
        generate_sdf3_xml(app, platform, mapping, xml_file)

        throughput = run_sdf3(xml_file)

        out["F"] = np.array([
            -throughput,
            cost
        ])


#######################################################
############### XML generation for SDF3 ###############
#######################################################

def generate_sdf3_xml(app, platform, mapping, filename):  # platform -= list of PEs that create from alloc
    """
    Generate SDF3 XML in official format.
    """

    sdf3_el = ET.Element(
        "sdf3",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "type": "sdf",
            "version": "1.0",
            "xsi:noNamespaceSchemaLocation": "/mnt/d/SDF3/sdf3/sdf/xsd/sdf3-sdf.xsd"
        }
    )

    app_graph = ET.SubElement(sdf3_el, "applicationGraph", name=app.name)
    sdf_el = ET.SubElement(app_graph, "sdf", name=app.name, type="SDF")

    arch_graph = ET.SubElement(sdf3_el, "architectureGraph", name="arch")
    for i, pe_type in enumerate(platform):
        tile = ET.SubElement(arch_graph, "tile", name=f"t{i}")
        ET.SubElement(tile, "processor", name=f"p{i}", type=pe_type)
        ET.SubElement(tile, "memory", name=f"m{i}", size="1024")  # dummy memory
        ET.SubElement(tile, "networkInterface", name=f"ni{i}")

    mapping_el = ET.SubElement(
        sdf3_el,
        "mapping",
        appGraph=app.name,
        archGraph="arch"
    )

    # group actors per tile
    tile_map = {}
    for task, pe_index in mapping.items():
        tile_name = f"t{pe_index}"
        tile_map.setdefault(tile_name, []).append(task)

    for tile_name, actors in tile_map.items():
        tile_el = ET.SubElement(mapping_el, "tile", name=tile_name)
        for actor_name in actors:
            ET.SubElement(tile_el, "actor", name=actor_name)

    # Actors
    for actor in app.actors.values():
        a_el = ET.SubElement(sdf_el, "actor", name=actor.name, type=actor.name)
        for port, rate in actor.in_ports.items():
            ET.SubElement(a_el, "port", name=port, type="in", rate=str(rate))
        for port, rate in actor.out_ports.items():
            ET.SubElement(a_el, "port", name=port, type="out", rate=str(rate))

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
            attrs["initialTokens"] = str(ch.init_tokens)
        ET.SubElement(sdf_el, "channel", **attrs)

    # sdfProperties
    sdf_props = ET.SubElement(app_graph, "sdfProperties")
    # Actor properties with execution times
    for actor in app.actors.values():
        actor_prop = ET.SubElement(sdf_props, "actorProperties", actor=actor.name)
        for pe_type, t in actor.exec_time.items():
            proc = ET.SubElement(actor_prop, "processor", type=pe_type, default="true")
            ET.SubElement(proc, "executionTime", time=str(t))

    # Channel properties
    for idx, ch in enumerate(app.channels, start=1):
        ch_name = f"ch{idx}"
        ET.SubElement(sdf_props, "channelProperties", channel=ch_name)

    ET.SubElement(sdf_props, "graphProperties")

    # Pretty print
    rough_string = ET.tostring(sdf3_el, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    pretty_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


###############################################
############### SDF3 invocation ###############
###############################################

def run_sdf3(xml_file):
    cmd = [
        "/mnt/d/SDF3/sdf3/build/release/Linux/bin/sdf3analysis-sdf",
        "--graph", xml_file,
        "--algo", "throughput"
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/mnt/d/SDF3/sdf3/build/release/Linux/lib:" + env.get("LD_LIBRARY_PATH", "")

    try:
        output = subprocess.check_output(cmd, env=env).decode()
        return parse_throughput(output)
    except subprocess.CalledProcessError as e:
        print("Error executing sdf3:", e)
        print("Output:", e.output.decode() if e.output else "")
        return 0.0


def parse_throughput(output):
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("thr("):
            return float(line.split("=")[-1].strip())
    return 0.0


####################################################
################# Operators ########################
####################################################


################################################################################
#                   Sampling for flexible chromosome                           #
################################################################################
class FlexibleSampling(Sampling):
    """
    Sampling for flexible chromosomes
    - allocation: random integers in [1..max_alloc]
    - binding: random integers in [0..P-1] where P = sum(allocation)
    """

    def __init__(self, n_types, n_tasks, max_alloc=4):
        super().__init__()
        self.n_types = n_types
        self.n_tasks = n_tasks
        self.max_alloc = int(max_alloc)

    def _do(self, problem, n_samples, **kwargs):
        pop = []
        for _ in range(n_samples):
            alloc = np.random.randint(1, self.max_alloc + 1, size=self.n_types)
            P = int(np.sum(alloc))
            bind = np.random.randint(0, P, size=self.n_tasks)
            chrom = np.concatenate([alloc, bind]).astype(int)
            pop.append(chrom)
        return np.array(pop)


################################################################################
# Crossover: two-stage (allocation crossover then binding inheritance + repair)#
################################################################################
class FlexibleCrossover(Crossover):
    """
    Two-stage crossover:
    1) crossover allocation vectors (one-point)
    2) for each task, randomly inherit binding from one of the parents, then repair binding modulo P
    """

    def __init__(self, n_types, n_tasks, prob=0.8):
        super().__init__(2, 2)
        self.n_types = n_types
        self.n_tasks = n_tasks
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        n_off, n_mat, n_var = X.shape
        Y = np.full_like(X, np.nan)

        for k in range(n_mat):
            p1 = X[0, k].copy().astype(int)
            p2 = X[1, k].copy().astype(int)
            c1 = p1.copy()
            c2 = p2.copy()

            if np.random.rand() < self.prob:
                # allocation one-point crossover
                if self.n_types > 1:
                    point = np.random.randint(1, self.n_types)
                else:
                    point = 1
                c1[:point] = p1[:point]
                c1[point:self.n_types] = p2[point:self.n_types]
                c2[:point] = p2[:point]
                c2[point:self.n_types] = p1[point:self.n_types]

                # binding inheritance per-task
                for j in range(self.n_tasks):
                    if np.random.rand() < 0.5:
                        c1[self.n_types + j] = p1[self.n_types + j]
                        c2[self.n_types + j] = p2[self.n_types + j]
                    else:
                        c1[self.n_types + j] = p2[self.n_types + j]
                        c2[self.n_types + j] = p1[self.n_types + j]

                # repair binding according to new allocations
                for child in (c1, c2):
                    alloc_child = child[:self.n_types].astype(int)
                    P = int(np.sum(alloc_child))
                    if P <= 0:
                        P = 1
                    for j in range(self.n_tasks):
                        child[self.n_types + j] = int(child[self.n_types + j]) % P
            else:
                c1 = p1
                c2 = p2

            Y[0, k] = c1
            Y[1, k] = c2

        return Y


################################################################################
#                  Mutation: multi-mode mutation                               #
################################################################################
class FlexibleMutation(Mutation):
    """
    Mutation supports:
    - allocation increment/decrement (change counts)
    - addPE / removePE implemented as increment/decrement of allocation counts
    - random reassign tasks (binding mutation)
    - bus-aware reassign (tasks moved between PEs on same bus)
    - heuristic placeholder (if deadlines provided)
    """

    def __init__(self, n_types, n_tasks, mutation_rate=0.6, max_alloc=4):
        super().__init__()
        self.n_types = n_types
        self.n_tasks = n_tasks
        self.mrate = float(mutation_rate)
        self.max_alloc = int(max_alloc)

    def _do(self, problem, X, **kwargs):
        Y = X.copy().astype(int)
        for i in range(Y.shape[0]):
            ind = Y[i]

            # pick which mutation group to run (allow multiple small changes)
            # allocation changes
            for t in range(self.n_types):
                if np.random.rand() < (self.mrate / 2):
                    # increase or decrease allocation count
                    delta = np.random.choice([-1, 1])
                    ind[t] = int(max(1, min(self.max_alloc, ind[t] + delta)))

            # optional addPE (increment some allocation)
            if np.random.rand() < (self.mrate * 0.4):
                idx = np.random.randint(0, self.n_types)
                ind[idx] = int(min(self.max_alloc, ind[idx] + 1))

            # optional removePE (decrement some allocation if >1)
            if np.random.rand() < (self.mrate * 0.4):
                idx = np.random.randint(0, self.n_types)
                ind[idx] = int(max(1, ind[idx] - 1))

            # rebuild platform size and simple buses for binding mutation
            alloc = ind[:self.n_types]
            P = int(np.sum(alloc))
            if P <= 0:
                P = 1

            # binding mutation: random reassign some tasks
            for j in range(self.n_tasks):
                if np.random.rand() < self.mrate:
                    ind[self.n_types + j] = np.random.randint(0, P)

            # bus-aware reassign: move some tasks within same bus
            if np.random.rand() < (self.mrate * 0.5):
                # build simple buses: first half, second half
                buses = []
                if P == 1:
                    buses = [[0]]
                else:
                    half = max(1, P // 2)
                    buses = [list(range(0, half)), list(range(half, P))]
                # pick a bus with at least 1 PE
                bidx = np.random.randint(0, len(buses))
                src_bus = buses[bidx]
                if len(src_bus) >= 2:
                    # choose a PE from this bus and a different PE in same bus
                    pe_from = np.random.choice(src_bus)
                    pe_to = np.random.choice([p for p in src_bus if p != pe_from])
                    # pick some tasks currently assigned to pe_from and move them
                    assigned_tasks = [j for j in range(self.n_tasks) if (ind[self.n_types + j] % P) == pe_from]
                    if assigned_tasks:
                        k = min(len(assigned_tasks), np.random.randint(1, 4))
                        to_move = np.random.choice(assigned_tasks, size=k, replace=False)
                        for tm in to_move:
                            ind[self.n_types + tm] = pe_to

            # heuristic reassignment placeholder (no deadlines implemented)
            # if problem has attribute task_deadlines (dict), we could implement
            # moving tasks that miss deadlines to less loaded PEs.
            if hasattr(problem, "task_deadlines") and np.random.rand() < (self.mrate * 0.2):
                # basic heuristic: move one random task to least loaded PE
                loads = [0] * P
                for j in range(self.n_tasks):
                    loads[ind[self.n_types + j] % P] += 1
                least = int(np.argmin(loads))
                tmove = np.random.randint(0, self.n_tasks)
                ind[self.n_types + tmove] = least

        return Y


#####################################################  main  #################################################

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

if __name__ == "__main__":

    problem = FlexibleArchProblem()

    algorithm = NSGA2(
        pop_size=20,
        sampling=FlexibleSampling(problem.n_types, problem.n_tasks, problem.max_alloc),
        crossover=FlexibleCrossover(problem.n_types, problem.n_tasks, prob=0.9),
        mutation=FlexibleMutation(problem.n_types, problem.n_tasks, mutation_rate=0.4, max_alloc=problem.max_alloc),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 5)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True
    )

    print("Best Solutions (Pareto Front):")
    for f in res.F:
        print("Throughput:", -f[0], "Cost:", f[1])