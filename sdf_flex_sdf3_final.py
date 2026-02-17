import numpy as np
import subprocess
from pymoo.core.problem import ElementwiseProblem
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

############### Application / Architecture models ###############

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

def generate_sdf3_xml(app, arch, mapping, filename):
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


############### SDF3 invocation ###############

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
        if "throughput" in line.lower():
            return float(line.split()[-1])
    return 0.0


############### Problem Definition ###############

class SDFMappingProblem(ElementwiseProblem):
    def __init__(self, application, pe_types, pe_cost, max_alloc_per_type=4, **kwargs):
        self.app = application
        self.actors = list(application.actors.keys())
        self.n_tasks = len(self.actors)

        self.pe_types = pe_types
        self.pe_cost = pe_cost
        self.n_types = len(pe_types)
        self.max_alloc = max_alloc_per_type

        n_var = self.n_types + self.n_tasks
        super().__init__(n_var=n_var, n_obj=2, elementwise=True, **kwargs)

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


############### TEST BENCH ########################

if __name__ == "__main__":

    # Application
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

    # Problem
    problem = SDFMappingProblem(
        application=app,
        pe_types=["gpp", "fpga"],
        pe_cost={"gpp": 50, "fpga": 100},
        max_alloc_per_type=3
    )

    # Fake chromosome
    alloc = np.array([2, 1])
    binding = np.array([0, 2, 1])
    x = np.concatenate([alloc, binding])

    # Evaluate
    out = {}
    problem._evaluate(x, out)

    # Print results
    print("\n========== CHROMOSOME ==========")
    print("Allocation:", alloc)
    print("Binding:", binding)

    print("\n========== OBJECTIVES ==========")
    print("F =", out["F"])

    # Show XML
    print("\n========== GENERATED XML ==========")
    with open("tmp_sdf_model.xml", "r", encoding="utf-8") as f:
        print(f.read())
