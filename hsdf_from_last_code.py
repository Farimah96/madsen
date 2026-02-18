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

################################################################################
#                              Helper functions                                #
################################################################################

def generate_sdf3_xml(app, arch, mapping, filename):

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

    # ------------------------------------------------------------------
    # Actors
    # ------------------------------------------------------------------
    for actor in app.actors.values():

        a_el = ET.SubElement(sdf_el, "actor", name=actor.name, type=actor.name)

        for port_name, rate in actor.in_ports.items():
            ET.SubElement(a_el, "port", name=port_name, type="in", rate=str(rate))

        for port_name, rate in actor.out_ports.items():
            ET.SubElement(a_el, "port", name=port_name, type="out", rate=str(rate))

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # SDF Properties
    # ------------------------------------------------------------------
    sdf_props = ET.SubElement(app_graph, "sdfProperties")

    # Actor execution times + optional mapping
    for actor in app.actors.values():

        actor_prop = ET.SubElement(sdf_props, "actorProperties", actor=actor.name)

        for pe_type, t in actor.exec_time.items():
            proc = ET.SubElement(actor_prop, "processor",
                                 type=pe_type,
                                 default="true")
            ET.SubElement(proc, "executionTime", time=str(t))

        # optional binding (only if mapping is provided)
        if mapping is not None and actor.name in mapping:
            ET.SubElement(actor_prop, "processorMapping",
                          processor=str(mapping[actor.name]))

    # Channel properties
    for idx, ch in enumerate(app.channels, start=1):
        ch_name = f"ch{idx}"
        ET.SubElement(sdf_props, "channelProperties", channel=ch_name)

    ET.SubElement(sdf_props, "graphProperties")

    # ------------------------------------------------------------------
    # Pretty formatting
    # ------------------------------------------------------------------
    rough_string = ET.tostring(sdf3_el, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    pretty_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml)




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
    except subprocess.CalledProcessError:
        return 0.0


def parse_throughput(output):
    for line in output.splitlines():
        if "throughput" in line.lower():
            parts = line.split()
            for p in parts:
                try:
                    return float(p)
                except:
                    continue
    return 0.0



def build_sdf_app(tasks, exec_time_table, pe_types, weights, tokens_matrix):

    app = FlexibleArchProblem.SDFApplication(name="my_app")

    # -------------------------------------------------
    # Step 1: Create actors with correct in/out ports
    # -------------------------------------------------
    for t_index, task in enumerate(tasks):

        exec_time = {
            pe_types[p_index]: exec_time_table[t_index][p_index]
            for p_index in range(len(pe_types))
        }

        actor = FlexibleArchProblem.Actor(name=task, exec_time=exec_time)

        # Count incoming edges
        in_count = 0
        for src_index in range(len(tokens_matrix)):
            if t_index < len(tokens_matrix[src_index]):
                has_edge, _ = tokens_matrix[src_index][t_index]
                if has_edge:
                    in_count += 1

        # Count outgoing edges
        out_count = 0
        if t_index < len(tokens_matrix):
            for dst_index in range(len(tokens_matrix[t_index])):
                has_edge, _ = tokens_matrix[t_index][dst_index]
                if has_edge:
                    out_count += 1

        actor.in_ports = {f"in{i}": 1 for i in range(in_count)}
        actor.out_ports = {f"out{i}": 1 for i in range(out_count)}

        app.actors[task] = actor

    # -------------------------------------------------
    # Step 2: Create channels using unique ports
    # -------------------------------------------------
    out_port_counter = {task: 0 for task in tasks}
    in_port_counter = {task: 0 for task in tasks}

    for src_index, src in enumerate(tasks):
        for dst_index, dst in enumerate(tasks):

            if (src_index < len(tokens_matrix) and
                dst_index < len(tokens_matrix[src_index])):

                has_edge, init_tokens = tokens_matrix[src_index][dst_index]

                if has_edge:
                    out_idx = out_port_counter[src]
                    in_idx = in_port_counter[dst]

                    ch = FlexibleArchProblem.Channel(
                        src_actor=src,
                        src_port=f"out{out_idx}",
                        dst_actor=dst,
                        dst_port=f"in{in_idx}",
                        init_tokens=init_tokens
                    )

                    app.channels.append(ch)

                    out_port_counter[src] += 1
                    in_port_counter[dst] += 1

    return app



################################################################################
#                              Problem definition                              #
################################################################################

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
        self.pe_cost = pe_cost if pe_cost is not None else {"fpga":100, "gpp":50, "asic":150, "dsp":80}

        if exec_time_table is not None:
            self.exec_time_table = exec_time_table
        else:
            self.exec_time_table = np.array([
                [0.9, 1.4, 0.7, 1],
                [1.1, 1.0, 0.6, 1.1],
                [0.8, 1.2, 0.9, 0.7],
                [1.3, 0.9, 0.7, 1.2],
            ], dtype=float)

        self.max_alloc = int(max_alloc_per_type)

        self.tokens_matrix = [
            [(False, 0),(True, 1),(True, 1),(False, 0)],
            [(False, 0),(False, 0),(True, 1),(True, 1)],
            [(False, 0),(False, 0),(False, 0),(True, 1)],
            [(True, 1),(False, 0),(False, 0),(False, 0)]
        ]

        n_var = self.n_types + self.n_tasks
        super().__init__(elementwise=elementwise, n_var=n_var, n_obj=2, **kwargs)

    def allocation_to_platform(self, alloc_vector):
        platform = []
        for t_index, cnt in enumerate(alloc_vector):
            if cnt is None:
                continue
            for _ in range(int(max(0, cnt))):
                platform.append(self.pe_types[t_index])
        return platform

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

        app = build_sdf_app(
            tasks=self.tasks,
            exec_time_table=self.exec_time_table,
            pe_types=self.pe_types,
            weights=self.weights,
            tokens_matrix=self.tokens_matrix
        )

        filename = "/tmp/current_chromosome.xml"
        generate_sdf3_xml(app, arch=None, mapping=None, filename=filename)

        throughput = run_sdf3(filename)

        cost = 0
        pes = list(mapping.values())
        del_dup = list(dict.fromkeys(pes))
        for p in del_dup:
            if p < 0 or p >= len(platform):
                continue
            ptype = platform[p]
            cost += self.pe_cost.get(ptype, 0)

        out["F"] = np.array([-throughput, cost])


#################################################
#################################################
################## main #########################
#################################################
#################################################

class FlexibleSampling(Sampling):
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
            if P <= 0:
                P = 1
            bind = np.random.randint(0, P, size=self.n_tasks)
            chrom = np.concatenate([alloc, bind]).astype(int)
            pop.append(chrom)
        return np.array(pop)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    problem = FlexibleArchProblem()

    sampling = FlexibleSampling(n_types=problem.n_types,
                                n_tasks=problem.n_tasks,
                                max_alloc=3)

    n_samples = 5
    population = sampling._do(problem, n_samples)

    print("=== Test evaluation for sampled chromosomes ===")
    for i, chrom in enumerate(population):
        out = {}
        problem._evaluate(chrom, out)
        print(f"Chromosome {i}: {chrom} -> Objectives (throughput, cost): {out['F']}")
