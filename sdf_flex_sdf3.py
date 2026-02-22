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

        xl = np.zeros(n_var)

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

        del_dup = list(dict.fromkeys(pes))
        for p in del_dup:
            if p < 0 or p >= len(platform):
                continue
            ptype = platform[p]
            cost += self.pe_cost.get(ptype, 0)  ############## 1 ?

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


#####################################################  main  #################################################

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

if __name__ == "__main__":

    problem = FlexibleArchProblem()

    algorithm = NSGA2(
        pop_size=5
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