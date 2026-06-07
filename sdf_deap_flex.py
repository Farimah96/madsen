
"""
Multi-Objective Hardware/Software Co-Design for SDF Applications using NSGA-II
================================================================================

Before running this script, make sure to set up the SDF3 environment:

    export LD_LIBRARY_PATH=/Your path/sdf3/build/release/Linux/lib:$LD_LIBRARY_PATH
    export PATH=$PATH:/Your path/sdf3/build/release/Linux/bin

Or add these lines to your ~/.bashrc file for permanent setup.

This code automatically explores and optimizes heterogeneous platform architectures 
for Synchronous Data Flow (SDF) applications using a custom evolutionary algorithm.

Platform & Environment:
    - OS: WSL2 Ubuntu 22.04 (Windows Subsystem for Linux)
    - IDE: Visual Studio Code with Python extension
    - Python: 3.8+ with pip packages (deap, numpy, matplotlib, scoop)
    - External Tool: SDF3 (compiled for Linux, requires proper library paths)

Architecture Exploration:
    - 4 Tasks: a, b, c, d (pipelined SDF application with feedback loop)
    - 4 PE Types: FPGA, GPP, ASIC, DSP (each with different cost and execution time)
    - Custom chromosome encoding:
        * Allocation Vector (4 integers): Number of each PE type [1..4]
        * Binding Vector (4 integers): Task-to-processor mapping (0 to P-1)
    - Flexible operators: Two-stage crossover + Multi-mode mutation with bus-aware relocation

Optimization Objectives (Maximize/Minimize):
    1. Maximize Throughput (samples/sec) - computed via SDF3 analysis tool
    2. Minimize Total Cost - sum(allocated_PEs[i] * cost_per_type[i])
    3. [Extensible] Minimize End-to-End Latency between specified actors (a → c)

SDF Application Graph Properties:
    - Consistent graph with repetition vector r = [2, 2, 2, 2] (after fixing initial tokens)
    - Rates matrix defines production/consumption rates between all actor pairs
    - Feedback channel (d → a) contains initialTokens=2 to prevent deadlock
    - All initial tokens properly placed in <channel> tags (not in actor ports)

Key Features:
    - Integration with SDF3 tool for accurate throughput analysis
    - Automatic XML generation following SDF3 schema standards
    - NSGA-II algorithm for Pareto front optimization (max throughput, min cost)
    - Parallel evaluation using SCOOP framework for faster convergence
    - Real-time visualization of evolution with generation-by-generation plots
    - Automatic cleanup of temporary XML files after each evaluation

Requirements Installation (WSL2 Ubuntu):
    $ sudo apt update
    $ sudo apt install python3-pip python3-tk
    $ pip install numpy deap matplotlib scoop

SDF3 Tool Setup:
    - Clone/build SDF3 from: https://github.com/sdfteam/sdf3
    - Path configured in run_sdf3() function (adjust for your system)
    - Ensure sdf3analysis-sdf binary has execute permissions
    - Test manually: sdf3analysis-sdf --graph test.xml --algo throughput

Output Directories (auto-created each run):
    - sdf_xml_files/         : Temporary XML files (cleaned after each run)
    - sdf_evolution_plot/    : Generation-by-generation scatter plots (PNG format)
        * Shows full population (blue circles) and Pareto front (red stars)
        * Includes best throughput/cost annotations on each plot
    
Evolution Parameters (configurable in main()):
    - Population size: 30 individuals
    - Generations: 20
    - Tournament size (MU): 100
    - Crossover probability: 0.9
    - Mutation probability: 0.5
    - Allocation mutation rate: 30% per gene
    - Binding mutation rate: 40% per gene

Algorithm Operators:
    - Crossover: Single-point crossover for allocation + uniform crossover for binding
    - Mutation: Increment/decrement allocation (±1) + random rebinding to valid processors
    - Selection: NSGA-II non-dominated sorting with crowding distance
    - Repair: Ensures binding indices remain within allocated processor count

Troubleshooting Common Issues:
    1. SDF3 not found: Check PATH and executable permissions in /mnt/d/SDF3/...
    2. Throughput = 0: Check for deadlock (initial tokens in feedback channel)
    3. LD_LIBRARY_PATH error: Ensure SDF3 lib path is set before running
    4. SCOOP parallel error: Fall back to standard map (remove futures.map registration)
    5. XML schema validation: Verify rates matrix has correct (prod,cons,init) tuples

CSV Logging (extensible - not yet implemented in current version):
    - evolution_data/population_genX.csv : Full population fitness values
    - evolution_data/pareto_front_genX.csv : Non-dominated solutions per generation
    - evolution_data/summary_stats.csv : Best/worst/avg metrics over time


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

        self.pe_cost = {"fpga":100, "gpp":50, "asic":150, "dsp":80}

        self.exec_time_table = np.array([
            [1, 2, 1, 3],
            [2, 1, 3, 1],
            [1, 1, 2, 2],
            [3, 2, 2, 1],
        ], dtype=float)

        # SDF rates matrix: (production_rate, consumption_rate, initial_tokens)
        self.rates_matrix = [
            #   a              b              c              d
            [(0,0,0),   (2,1,0),  (2,1,0),  (0,0,0)],   # a -> b: prod=2, cons=1 
            [(0,0,0),   (0,0,0),  (2,2,0),  (2,2,0)],
            [(0,0,0),   (0,0,0),  (0,0,0),  (2,2,0)],   
            [(1,2,2),   (0,0,0),  (0,0,0),  (0,0,0)]    # d -> a: prod=1, cons=2, init_tokens=1
        ]

        self.max_alloc = 4
        
    def allocation_to_platform(self, alloc_vector):
        """Convert allocation vector to platform list of PE types"""
        platform = []
        for t_index, cnt in enumerate(alloc_vector):
            for _ in range(int(max(1, cnt))):
                platform.append(self.pe_types[t_index])
        return platform

    def binding_to_mapping(self, platform, binding):
        """Convert binding genes to task-to-processor mapping"""
        mapping = {}
        P = len(platform)
        if P == 0:
            P = 1
        for i, task in enumerate(self.tasks):
            mapping[task] = int(binding[i]) % P
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
            "xsi:noNamespaceSchemaLocation": "/mnt/d/SDF3/sdf3/sdf/xsd/sdf3-sdf.xsd"
        }
    )

    app_graph = ET.SubElement(sdf3_el, "applicationGraph", name=app.name)
    sdf_el = ET.SubElement(app_graph, "sdf", name=app.name, type="SDF")
    arch_graph = ET.SubElement(sdf3_el, "architectureGraph", name="arch")
    
    # Architecture tiles
    for i, pe_type in enumerate(platform):
        tile = ET.SubElement(arch_graph, "tile", name=f"t{i}")
        ET.SubElement(tile, "processor", name=f"p{i}", type=pe_type)
        ET.SubElement(tile, "memory", name=f"m{i}", size="1024")
        ET.SubElement(tile, "networkInterface", name=f"ni{i}")

    # Mapping
    mapping_el = ET.SubElement(sdf3_el, "mapping", appGraph="app", archGraph="arch")
    tile_map = {}
    for task, pe_index in mapping.items():
        tile_name = f"t{pe_index}"
        tile_map.setdefault(tile_name, []).append(task)

    for tile_name, actors in tile_map.items():
        tile_el = ET.SubElement(mapping_el, "tile", name=tile_name)
        for actor_name in actors:
            ET.SubElement(tile_el, "actor", name=actor_name)

    # Actors with ports
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

    # Properties
    sdf_props = ET.SubElement(app_graph, "sdfProperties")
    
    # Actor properties with execution times
    for actor in app.actors.values():
        actor_prop = ET.SubElement(sdf_props, "actorProperties", actor=actor.name)
        
        assigned_processor_type = None
        for pe_index, pe_type in enumerate(platform):
            if mapping.get(actor.name) == pe_index:
                assigned_processor_type = pe_type
                break

        if assigned_processor_type and assigned_processor_type in actor.exec_time:
            proc = ET.SubElement(actor_prop, "processor", type=assigned_processor_type, default="true")
            ET.SubElement(proc, "executionTime", time=str(actor.exec_time[assigned_processor_type]))
        else:
            first_type = list(actor.exec_time.keys())[0] if actor.exec_time else "gpp"
            proc = ET.SubElement(actor_prop, "processor", type=first_type, default="true")
            ET.SubElement(proc, "executionTime", time=str(actor.exec_time.get(first_type, 1)))

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


def run_sdf3(xml_file):
    """Run SDF3 throughput analysis on XML file"""
    cmd = [
        "/mnt/d/SDF3/sdf3/build/release/Linux/bin/sdf3analysis-sdf",
        "--graph", xml_file,
        "--algo", "throughput"
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/mnt/d/SDF3/sdf3/build/release/Linux/lib:" + env.get("LD_LIBRARY_PATH", "")

    try:
        output = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT).decode()
        return parse_throughput(output)
    except subprocess.CalledProcessError as e:
        print(f"SDF3 error: {e.output.decode() if e.output else 'Unknown error'}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error: {e}")
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

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()


def init_individual():
    """Initialize a random individual"""
    alloc = np.random.randint(1, problem.max_alloc + 1, size=problem.n_types)
    P = int(np.sum(alloc))
    bind = np.random.randint(0, P, size=problem.n_tasks)
    return creator.Individual(list(np.concatenate([alloc, bind])))


toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ==============================
# Custom Crossover
# ==============================

def custom_crossover(ind1, ind2):
    """Custom crossover for allocation-binding individuals"""
    
    n_types = problem.n_types
    n_tasks = problem.n_tasks

    # Crossover allocation part
    point = random.randint(1, n_types-1)
    ind1[:point], ind2[:point] = ind2[:point], ind1[:point]

    # Crossover binding part
    for j in range(n_tasks):
        if random.random() < 0.5:
            ind1[n_types+j], ind2[n_types+j] = ind2[n_types+j], ind1[n_types+j]

    # Repair individuals
    for ind in (ind1, ind2):
        alloc = ind[:n_types]
        P = sum(alloc)
        if P <= 0:
            P = 1
        for j in range(n_tasks):
            ind[n_types+j] %= P

    return ind1, ind2


# ==============================
# Custom Mutation
# ==============================

def custom_mutation(ind):
    """Custom mutation for allocation-binding individuals"""
    
    n_types = problem.n_types
    n_tasks = problem.n_tasks

    # Mutate allocation
    for i in range(n_types):
        if random.random() < 0.3:
            delta = random.choice([-1, 1])
            ind[i] = max(1, min(problem.max_alloc, ind[i] + delta))

    # Mutate binding
    P = sum(ind[:n_types])
    if P <= 0:
        P = 1

    for j in range(n_tasks):
        if random.random() < 0.4:
            ind[n_types+j] = random.randint(0, P-1)

    return ind,


toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutation)
toolbox.register("select", tools.selNSGA2)


# ==============================
# Evaluation 
# ==============================


def evaluate(individual):
    """Evaluate individual: throughput and cost"""
    
    alloc = individual[:problem.n_types]
    binding = individual[problem.n_types:]
    
    platform = problem.allocation_to_platform(alloc)
    mapping = problem.binding_to_mapping(platform, binding)
    
    # Build SDF application
    app = SDFApplication("app")
    
    # Create actors with execution times
    for i, task in enumerate(problem.tasks):
        exec_times = {}
        for j, pe_type in enumerate(problem.pe_types):
            exec_times[pe_type] = problem.exec_time_table[i][j]
        actor = Actor(task, exec_times)
        app.actors[task] = actor

    # Create channels from rates_matrix
    for i in range(problem.n_tasks):
        for j in range(problem.n_tasks):
            prod_rate, cons_rate, init_tokens = problem.rates_matrix[i][j]
            if prod_rate > 0 and cons_rate > 0:
                src_actor = problem.tasks[i]
                dst_actor = problem.tasks[j]
                src_port = f"out_{i}_{j}"
                dst_port = f"in_{i}_{j}"
                channel = Channel(src_actor, src_port, dst_actor, dst_port, init_tokens)
                app.channels.append(channel)
                app.actors[src_actor].out_ports[src_port] = prod_rate
                app.actors[dst_actor].in_ports[dst_port] = cons_rate
    
    # Generate XML and run SDF3
    xml_filename = f"tmp_{random.randint(1, 1000000)}.xml"
    full_xml_path = os.path.join(XML_OUTPUT_DIR, xml_filename)
    generate_sdf3_xml(app, platform, mapping, full_xml_path)
    
    throughput = run_sdf3(full_xml_path)
    cost = sum(alloc[i] * problem.pe_cost[problem.pe_types[i]] for i in range(problem.n_types))
    
    # Clean up temporary file
    try:
        os.remove(full_xml_path)
        # pass
    except:
        pass
    
    return throughput, cost


toolbox.register("evaluate", evaluate)

# Register SCOOP's parallel map function
# This replaces Python's default map with distributed version
toolbox.register("map", futures.map)  # ADD THIS LINE

# ==============================
# Plot Utilities
# ==============================

def save_evolution_plot(population, generation):
    """Save plot for each generation with different colors for Pareto front"""
    
    os.makedirs(EVOLUTION_PLOTS_DIR, exist_ok=True)
    
    valid_individuals = [ind for ind in population if ind.fitness.valid]
    if not valid_individuals:
        return
    
    throughputs = [ind.fitness.values[0] for ind in valid_individuals]
    costs = [ind.fitness.values[1] for ind in valid_individuals]
    
    # Find Pareto front
    non_dominated = tools.sortNondominated(valid_individuals, k=len(valid_individuals), first_front_only=True)[0]
    nd_throughputs = [ind.fitness.values[0] for ind in non_dominated]
    nd_costs = [ind.fitness.values[1] for ind in non_dominated]
    
    # Sort Pareto front for better visualization
    sorted_pairs = sorted(zip(nd_throughputs, nd_costs))
    nd_throughputs_sorted, nd_costs_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Draw population
    plt.scatter(throughputs, costs, alpha=0.6, c='blue', s=50, 
               label=f'Population (n={len(valid_individuals)})', edgecolors='black', linewidth=0.5)
    
    # Draw Pareto front
    if nd_throughputs_sorted:
        plt.scatter(nd_throughputs_sorted, nd_costs_sorted, c='red', s=150, 
                   marker='*', label=f'Pareto Front (n={len(nd_throughputs_sorted)})', 
                   edgecolors='darkred', linewidth=1.5, zorder=5)
        
        # Connection line for Pareto front
        plt.plot(nd_throughputs_sorted, nd_costs_sorted, 'r--', alpha=0.5, linewidth=1)
    
    plt.xlabel("Throughput", fontsize=12, fontweight='bold')
    plt.ylabel("Cost", fontsize=12, fontweight='bold')
    plt.title(f"Generation {generation} - Population vs Pareto Front", fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add info text
    if throughputs:
        info_text = f"Best Throughput: {max(throughputs):.4f}\nBest Cost: {min(costs):.2f}"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = f"{EVOLUTION_PLOTS_DIR}/generation_{generation:03d}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================
# Main Evolution Loop
# ==============================

def main():

    cleanup_old_xml_files()
    
    pop = toolbox.population(n=30)
    NGEN = 20
    MU = 100
    CXPB = 0.9
    MUTPB = 0.5

    # Initial evaluation
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Save initial generation plot
    save_evolution_plot(pop, 0)

    # Evolution loop
    for gen in range(1, NGEN + 1):
        
        # Generate offspring
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        
        # Evaluate offspring
        fitnesses = list(toolbox.map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        
        # Select next generation
        pop = toolbox.select(pop + offspring, MU)
        
        # Save plot for current generation
        save_evolution_plot(pop, gen)
        
        # Print progress
        best_throughput = max(ind.fitness.values[0] for ind in pop if ind.fitness.valid)
        best_cost = min(ind.fitness.values[1] for ind in pop if ind.fitness.valid)
        print(f"Generation {gen}: Best Throughput={best_throughput:.4f}, Best Cost={best_cost:.2f}")

    # Display final Pareto front
    front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    print("\n" + "="*50)
    print("Final Pareto Front:")
    print("="*50)
    for i, ind in enumerate(sorted(front, key=lambda x: x.fitness.values[0], reverse=True)):
        print(f"{i+1}. Throughput: {ind.fitness.values[0]:.6f} | Cost: {ind.fitness.values[1]:.2f}")
        
    

if __name__ == "__main__":
    main()