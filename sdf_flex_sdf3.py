########################Application Model
class Actor:
    def __init__(self, name):
        self.name = name
        self.exec_time = {}      # {pe_type: time}
        self.in_ports = {}       # {port_name: token_rate}
        self.out_ports = {}      # {port_name: token_rate}



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




#########################Architecture Model

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




arch = Architecture()
arch.add_pe(ProcessingElement("P1", "gpp", 50))
arch.add_pe(ProcessingElement("P2", "fpga", 100))




#######################  Model → XML   #######################

import xml.etree.ElementTree as ET

def generate_sdf3_xml(app, arch, filename):
    sdf = ET.Element("sdf")


#################### Application → XML

app_el = ET.SubElement(sdf, "application", name=app.name)

for actor in app.actors.values():
    a_el = ET.SubElement(app_el, "actor", name=actor.name)

    for pe, t in actor.exec_time.items():
        ET.SubElement(
            a_el,
            "executionTime",
            processor=pe,
            value=str(t)
        )

    for port, rate in actor.in_ports.items():
        ET.SubElement(a_el, "inPort", name=port, rate=str(rate))

    for port, rate in actor.out_ports.items():
        ET.SubElement(a_el, "outPort", name=port, rate=str(rate))


###################### Channel → XML

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



############################ Architecture → XML

arch_el = ET.SubElement(sdf, "architecture")

for pe in arch.pes:
    ET.SubElement(
        arch_el,
        "processor",
        name=pe.name,
        type=pe.type,
        cost=str(pe.cost)
    )


tree = ET.ElementTree(sdf)
tree.write(filename, encoding="utf-8", xml_declaration=True)




######################_evaluate

generate_sdf3_xml(app, arch, "model.xml")

cmd = [
    "sdf3analysis-sdf",
    "--graph", "model.xml",
    "--algo", "throughput"
]

result = subprocess.check_output(cmd).decode()
throughput = parse_output(result)



############ mapping in eval #################

def chromosome_to_mapping(chromosome, actors, processors):
    mapping = {}
    for i, gene in enumerate(chromosome):
        mapping[actors[i]] = processors[gene]
    return mapping
