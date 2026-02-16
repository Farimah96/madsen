import numpy as np
import subprocess
from pymoo.core.problem import ElementwiseProblem
import os


############### Application / Architecture models (used as-is) ###############

class Actor:
    def __init__(self, name):
        self.name = name
        self.exec_time = {}
        self.in_ports = {}
        self.out_ports = {}


class Channel:
    def __init__(self, src_actor, src_port, dst_actor, dst_port,
                 prod_rate, cons_rate, init_tokens=0):
        self.src_actor = src_actor
        self.src_port = src_port
        self.dst_actor = dst_actor
        self.dst_port = dst_port
        self.prod_rate = prod_rate
        self.cons_rate = cons_rate
        self.init_tokens = init_tokens


class Application:
    def __init__(self, name):
        self.name = name
        self.actors = {}
        self.channels = []

    def add_actor(self, actor):
        self.actors[actor.name] = actor

    def add_channel(self, channel):
        self.channels.append(channel)


class ProcessingElement:
    def __init__(self, name, pe_type, cost):
        self.name = name
        self.type = pe_type
        self.cost = cost


class Architecture:
    def __init__(self):
        self.pes = []

    def add_pe(self, pe):
        self.pes.append(pe)


############### XML generation for SDF3 ###############

import xml.etree.ElementTree as ET

def generate_sdf3_xml(app, arch, mapping, filename):
    sdf = ET.Element("sdf")

    app_el = ET.SubElement(sdf, "application", name=app.name)

    for actor in app.actors.values():
        a_el = ET.SubElement(app_el, "actor", name=actor.name)

        for pe_type, t in actor.exec_time.items():
            ET.SubElement(
                a_el,
                "executionTime",
                processor=pe_type,
                value=str(t)
            )

        for port, rate in actor.in_ports.items():
            ET.SubElement(a_el, "inPort", name=port, rate=str(rate))

        for port, rate in actor.out_ports.items():
            ET.SubElement(a_el, "outPort", name=port, rate=str(rate))

    for ch in app.channels:
        ET.SubElement(
            app_el,
            "channel",
            srcActor=ch.src_actor,
            srcPort=ch.src_port,
            dstActor=ch.dst_actor,
            dstPort=ch.dst_port,
            prodRate=str(ch.prod_rate),
            consRate=str(ch.cons_rate),
            initialTokens=str(ch.init_tokens)
        )

    arch_el = ET.SubElement(sdf, "architecture")

    for pe in arch.pes:
        ET.SubElement(
            arch_el,
            "processor",
            name=pe.name,
            type=pe.type,
            cost=str(pe.cost)
        )

    map_el = ET.SubElement(sdf, "mapping")

    for actor, pe_name in mapping.items():
        ET.SubElement(
            map_el,
            "bind",
            actor=actor,
            processor=pe_name
        )

    tree = ET.ElementTree(sdf)
    tree.write(filename, encoding="utf-8", xml_declaration=True)


############### SDF3 invocation ###############


def win_to_wsl_path(win_path):
    win_path = os.path.abspath(win_path)
    drive = win_path[0].lower()
    path_rest = win_path[2:].replace("\\", "/")
    return f"/mnt/{drive}/{path_rest}"

def run_sdf3(xml_file_win):
    xml_file_wsl = win_to_wsl_path(xml_file_win)

    cmd = [
        "wsl",
        "/mnt/d/SDF3/sdf3/build/release/Linux/bin/sdf3analysis-sdf",
        "--graph", xml_file_wsl,
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
        if "throughput" in line.lower():
            return float(line.split()[-1])
    return 0.0


############### Problem Definition ###############

class SDFMappingProblem(ElementwiseProblem):

    def __init__(self,
                 application,
                 pe_types,
                 pe_cost,
                 max_alloc_per_type=4,
                 **kwargs):

        self.app = application
        self.actors = list(application.actors.keys())
        self.n_tasks = len(self.actors)

        self.pe_types = pe_types
        self.pe_cost = pe_cost
        self.n_types = len(pe_types)
        self.max_alloc = max_alloc_per_type

        n_var = self.n_types + self.n_tasks
        super().__init__(n_var=n_var, n_obj=2, elementwise=True, **kwargs)

    ############### allocation → Architecture ###############
    def allocation_to_architecture(self, alloc):
        arch = Architecture()
        idx = 0
        for t_index, cnt in enumerate(alloc):
            for _ in range(max(0, int(cnt))):
                pe = ProcessingElement(
                    name=f"P{idx}",
                    pe_type=self.pe_types[t_index],
                    cost=self.pe_cost[self.pe_types[t_index]]
                )
                arch.add_pe(pe)
                idx += 1
        return arch

    ############### binding → mapping ###############
    def binding_to_mapping(self, binding, arch):
        mapping = {}
        P = len(arch.pes)
        if P == 0:
            for actor in self.actors:
                mapping[actor] = "P0"
            return mapping

        for i, actor in enumerate(self.actors):
            pe = arch.pes[int(binding[i]) % P]
            mapping[actor] = pe.name
        return mapping

    ############### evaluate ###############
    def _evaluate(self, x, out, *args, **kwargs):

        alloc = x[:self.n_types].astype(int)
        binding = x[self.n_types:].astype(int)

        arch = self.allocation_to_architecture(alloc)
        mapping = self.binding_to_mapping(binding, arch)

        xml_file = "tmp_sdf_model.xml"
        generate_sdf3_xml(self.app, arch, mapping, xml_file)

        throughput = run_sdf3(xml_file)

        cost = sum(pe.cost for pe in arch.pes)

        out["F"] = np.array([
            -throughput,
            cost
        ])



############################################################
######################## TEST BENCH ########################
############################################################

if __name__ == "__main__":

    ######################## Application ########################
    app = Application("toy_sdf")

    a = Actor("A")
    a.exec_time = {"gpp": 3, "fpga": 1}
    a.out_ports = {"out": 1}

    b = Actor("B")
    b.exec_time = {"gpp": 2, "fpga": 1}
    b.in_ports = {"in": 1}
    b.out_ports = {"out": 1}

    c = Actor("C")
    c.exec_time = {"gpp": 4, "fpga": 2}
    c.in_ports = {"in": 1}

    app.add_actor(a)
    app.add_actor(b)
    app.add_actor(c)

    app.add_channel(Channel("A", "out", "B", "in", 1, 1))
    app.add_channel(Channel("B", "out", "C", "in", 1, 1))

    ######################## Problem ########################
    problem = SDFMappingProblem(
        application=app,
        pe_types=["gpp", "fpga"],
        pe_cost={"gpp": 50, "fpga": 100},
        max_alloc_per_type=3
    )

    ######################## Fake chromosome ########################
    alloc = np.array([2, 1])       # 2 GPP, 1 FPGA
    binding = np.array([0, 2, 1])  # A->P0, B->P2, C->P1
    x = np.concatenate([alloc, binding])

    ######################## Evaluate ########################
    out = {}
    problem._evaluate(x, out)

    ######################## Print results ########################
    print("\n========== CHROMOSOME ==========")
    print("Allocation:", alloc)
    print("Binding:", binding)

    print("\n========== OBJECTIVES ==========")
    print("F =", out["F"])

    ######################## Show XML ########################
    print("\n========== GENERATED XML ==========")
    with open("tmp_sdf_model.xml", "r") as f:
        print(f.read())
