import random
import numpy as np
import os
import xml.etree.ElementTree as ET
import subprocess
from xml.dom import minidom
from deap import base, creator, tools, algorithms

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import csv
import pandas as pd

# ==============================
# XML files organization
# ==============================

XML_OUTPUT_DIR = "xml_files"  # Directory for temporary XML files

def cleanup_old_xml_files():
    """Remove old XML files directory if it exists"""
    if os.path.exists(XML_OUTPUT_DIR):
        import shutil
        shutil.rmtree(XML_OUTPUT_DIR)
    os.makedirs(XML_OUTPUT_DIR, exist_ok=True)

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

        self.tokens_matrix = [
            [(False, 0),(True, 0),(True, 0),(False, 0)],
            [(False, 0),(False, 0),(True, 0),(True, 0)],
            [(False, 0),(False, 0),(False, 0),(True, 0)],
            [(True, 1),(False, 0),(False, 0),(False, 0)]
        ]

        self.max_alloc = 4


    def allocation_to_platform(self, alloc_vector):
        platform = []
        for t_index, cnt in enumerate(alloc_vector):
            # operator new: using max(1, cnt) to ensure at least one PE per type
            for _ in range(int(max(0, cnt))):  # operator new - fixed
                platform.append(self.pe_types[t_index])
        return platform


    def binding_to_mapping(self, platform, binding):
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
    
    for i, pe_type in enumerate(platform):
        tile = ET.SubElement(arch_graph, "tile", name=f"t{i}")
        ET.SubElement(tile, "processor", name=f"p{i}", type=pe_type)
        ET.SubElement(tile, "memory", name=f"m{i}", size="1024")
        ET.SubElement(tile, "networkInterface", name=f"ni{i}")

    mapping_el = ET.SubElement(sdf3_el, "mapping", appGraph="app", archGraph="arch")

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
            print(f"Warning: Processor type for actor '{actor.name}' not found")

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
    cmd = [
        "/mnt/d/SDF3/sdf3/build/release/Linux/bin/sdf3analysis-sdf",
        "--graph", xml_file,
        "--algo", "throughput"
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/mnt/d/SDF3/sdf3/build/release/Linux/lib:" + env.get("LD_LIBRARY_PATH", "")

    try:
        output = subprocess.check_output(cmd).decode()
        return parse_throughput(output)
    except:
        return 0.0


def parse_throughput(output):
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("thr("):
            val = line.split("=")[-1].strip()
            if val == "inf":
                return 1e6
            return float(val)
    return 0.0


# ==============================
# DEAP Setup
# ==============================

problem = FlexibleArchProblem()

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()


# operator new: Flexible Sampling
def flexible_sampling():
    """Sampling for flexible chromosomes
    - allocation: random integers in [1..max_alloc]
    - binding: random integers in [0..P-1] where P = sum(allocation)
    """
    # operator new - using max_alloc + 1 to include max_alloc value
    alloc = np.random.randint(1, problem.max_alloc + 1, size=problem.n_types)
    P = int(np.sum(alloc))
    bind = np.random.randint(0, P, size=problem.n_tasks)
    return creator.Individual(list(np.concatenate([alloc, bind])))


toolbox.register("individual", flexible_sampling)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# operator new: Flexible Crossover (two-stage)
def flexible_crossover(ind1, ind2):
    """
    Two-stage crossover:
    1) crossover allocation vectors (one-point)
    2) for each task, randomly inherit binding from one of the parents, then repair binding modulo P
    """
    n_types = problem.n_types
    n_tasks = problem.n_tasks
    
    c1 = ind1[:]
    c2 = ind2[:]
    
    if random.random() < 0.8:  # crossover probability
        # operator new - allocation one-point crossover
        if n_types > 1:
            point = random.randint(1, n_types - 1)
        else:
            point = 1
        
        # swap allocation parts
        c1[:point] = ind1[:point]
        c1[point:n_types] = ind2[point:n_types]
        c2[:point] = ind2[:point]
        c2[point:n_types] = ind1[point:n_types]
        
        # operator new - binding inheritance per-task
        for j in range(n_tasks):
            if random.random() < 0.5:
                c1[n_types + j] = ind1[n_types + j]
                c2[n_types + j] = ind2[n_types + j]
            else:
                c1[n_types + j] = ind2[n_types + j]
                c2[n_types + j] = ind1[n_types + j]
        
        # operator new - repair binding according to new allocations
        for child in (c1, c2):
            alloc_child = child[:n_types]
            P = int(np.sum(alloc_child))
            if P <= 0:
                P = 1
            for j in range(n_tasks):
                child[n_types + j] = int(child[n_types + j]) % P
    
    # copy back to original individuals
    ind1[:] = c1
    ind2[:] = c2
    
    return ind1, ind2


# operator new: Flexible Mutation (multi-mode)
def flexible_mutation(ind):
    """
    Mutation supports:
    - allocation increment/decrement (change counts)
    - addPE / removePE implemented as increment/decrement of allocation counts
    - random reassign tasks (binding mutation)
    - bus-aware reassign (tasks moved between PEs on same bus)
    """
    n_types = problem.n_types
    n_tasks = problem.n_tasks
    mutation_rate = 0.6  # operator new - mutation rate
    max_alloc = problem.max_alloc
    
    ind_copy = ind[:]
    
    # operator new - allocation changes (increment/decrement)
    for t in range(n_types):
        if random.random() < (mutation_rate / 2):
            delta = random.choice([-1, 1])
            ind_copy[t] = int(max(1, min(max_alloc, ind_copy[t] + delta)))
    
    # operator new - addPE (increment some allocation)
    if random.random() < (mutation_rate * 0.4):
        idx = random.randint(0, n_types - 1)
        ind_copy[idx] = int(min(max_alloc, ind_copy[idx] + 1))
    
    # operator new - removePE (decrement some allocation if >1)
    if random.random() < (mutation_rate * 0.4):
        idx = random.randint(0, n_types - 1)
        ind_copy[idx] = int(max(1, ind_copy[idx] - 1))
    
    # operator new - rebuild platform size for binding mutation
    alloc = ind_copy[:n_types]
    P = int(np.sum(alloc))
    if P <= 0:
        P = 1
    
    # operator new - binding mutation: random reassign some tasks
    for j in range(n_tasks):
        if random.random() < mutation_rate:
            ind_copy[n_types + j] = random.randint(0, P - 1)
    
    # operator new - bus-aware reassign: move some tasks within same bus
    if random.random() < (mutation_rate * 0.5):
        # build simple buses: first half, second half
        buses = []
        if P == 1:
            buses = [[0]]
        else:
            half = max(1, P // 2)
            buses = [list(range(0, half)), list(range(half, P))]
        
        # pick a bus with at least 1 PE
        bidx = random.randint(0, len(buses) - 1)
        src_bus = buses[bidx]
        if len(src_bus) >= 2:
            # choose a PE from this bus and a different PE in same bus
            pe_from = random.choice(src_bus)
            possible_pe_to = [p for p in src_bus if p != pe_from]
            if possible_pe_to:
                pe_to = random.choice(possible_pe_to)
                # pick some tasks currently assigned to pe_from and move them
                assigned_tasks = [j for j in range(n_tasks) if (ind_copy[n_types + j] % P) == pe_from]
                if assigned_tasks:
                    k = min(len(assigned_tasks), random.randint(1, 4))
                    to_move = random.sample(assigned_tasks, k)
                    for tm in to_move:
                        ind_copy[n_types + tm] = pe_to
    
    ind[:] = ind_copy
    return ind,


toolbox.register("mate", flexible_crossover)
toolbox.register("mutate", flexible_mutation)
toolbox.register("select", tools.selNSGA2)


# ==============================
# Evaluation
# ==============================

def evaluate(individual):
    alloc = individual[:problem.n_types]
    binding = individual[problem.n_types:]
    
    platform = problem.allocation_to_platform(alloc)
    mapping = problem.binding_to_mapping(platform, binding)
    
    app = SDFApplication("app")
    
    for i, task in enumerate(problem.tasks):
        exec_times = {}
        for j, pe_type in enumerate(problem.pe_types):
            exec_times[pe_type] = problem.exec_time_table[i][j]
        actor = Actor(task, exec_times)
        app.actors[task] = actor

    for i in range(problem.n_tasks):
        for j in range(problem.n_tasks):
            has_channel, init_tokens = problem.tokens_matrix[i][j]
            if has_channel:
                src_actor = problem.tasks[i]
                dst_actor = problem.tasks[j]
                src_port = f"out_{i}_{j}"
                dst_port = f"in_{i}_{j}"
                channel = Channel(src_actor, src_port, dst_actor, dst_port, init_tokens)
                app.channels.append(channel)
                app.actors[src_actor].out_ports[src_port] = 1
                app.actors[dst_actor].in_ports[dst_port] = 1
    
    xml_filename = f"tmp_{random.randint(1, 1_000_000)}.xml"
    
    generate_sdf3_xml(app, platform, mapping, xml_filename)
    full_xml_path = os.path.join(XML_OUTPUT_DIR, xml_filename)
    throughput = run_sdf3(full_xml_path)
    cost = sum(alloc[i] * problem.pe_cost[problem.pe_types[i]] for i in range(problem.n_types))
    
    # Clean up temporary file
    try:
        os.remove(full_xml_path)
    except:
        pass
    
    return throughput, cost


toolbox.register("evaluate", evaluate)


# ==============================
# Plot
# ==============================

def save_evolution_plot(population, generation, save_dir="evolution_plots"):
    """
    Save plot for each generation with different colors for Pareto front
    """
    os.makedirs(save_dir, exist_ok=True)
    
    valid_individuals = [ind for ind in population if ind.fitness.valid]
    if not valid_individuals:
        return
    
    throughputs = [ind.fitness.values[0] for ind in valid_individuals]
    costs = [ind.fitness.values[1] for ind in valid_individuals]
    
    non_dominated = tools.sortNondominated(valid_individuals, k=len(valid_individuals), first_front_only=True)[0]
    nd_throughputs = [ind.fitness.values[0] for ind in non_dominated]
    nd_costs = [ind.fitness.values[1] for ind in non_dominated]
    
    sorted_pairs = sorted(zip(nd_throughputs, nd_costs))
    nd_throughputs_sorted, nd_costs_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(throughputs, costs, alpha=0.6, c='blue', s=50, 
               label=f'Population (n={len(valid_individuals)})', edgecolors='black', linewidth=0.5)
    
    if nd_throughputs_sorted:
        plt.scatter(nd_throughputs_sorted, nd_costs_sorted, c='red', s=150, 
                   marker='*', label=f'Pareto Front (n={len(nd_throughputs_sorted)})', 
                   edgecolors='darkred', linewidth=1.5, zorder=5)
        plt.plot(nd_throughputs_sorted, nd_costs_sorted, 'r--', alpha=0.5, linewidth=1)
    
    plt.xlabel("Throughput", fontsize=12, fontweight='bold')
    plt.ylabel("Cost", fontsize=12, fontweight='bold')
    plt.title(f"Generation {generation} - Population vs Pareto Front", fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    info_text = f"Best Throughput: {max(throughputs):.4f}\nBest Cost: {min(costs):.2f}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = f"{save_dir}/generation_{generation:03d}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved: {filename}")

# ==============================
# Data Logging Utilities
# ==============================

def save_generation_data(population, generation, save_dir="evolution_data"):
    """
    Save throughput and cost of all individuals in a generation to CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract fitness values
    valid_individuals = [ind for ind in population if ind.fitness.valid]
    if not valid_individuals:
        return []
    
    # Prepare data
    data = []
    for idx, ind in enumerate(valid_individuals):
        throughput = ind.fitness.values[0]
        cost = ind.fitness.values[1]
        
        # Extract allocation and binding for additional info (optional)
        alloc = ind[:problem.n_types]
        binding = ind[problem.n_types:]
        
        data.append({
            'individual_id': idx,
            'throughput': throughput,
            'cost': cost,
            'allocation': str(list(alloc)),  # Fixed: convert to list first
            'binding': str(list(binding)),    # Fixed: convert to list first
            'total_pe_count': sum(alloc)
        })
    
    # Save to CSV
    filename = f"{save_dir}/generation_{generation:03d}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['individual_id', 'throughput', 'cost', 
                                               'allocation', 'binding', 'total_pe_count'])
        writer.writeheader()
        writer.writerows(data)
    
    # Also save a summary file with best solutions per generation
    summary_file = f"{save_dir}/summary.csv"
    best_throughput = max(data, key=lambda x: x['throughput'])
    best_cost = min(data, key=lambda x: x['cost'])
    
    # Check if summary file exists to append or create
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['generation', 'best_throughput', 'best_throughput_cost', 
                           'best_cost', 'best_cost_throughput', 'population_size'])
        writer.writerow([generation, best_throughput['throughput'], best_throughput['cost'],
                        best_cost['cost'], best_cost['throughput'], len(valid_individuals)])
    
    print(f"✓ Generation {generation} data saved: {filename}")
    return data

def save_all_generations_data(all_generations_data, save_dir="evolution_data"):
    """
    Save combined data from all generations to a single file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    combined_data = []
    for gen_idx, gen_data in enumerate(all_generations_data):
        if gen_data:
            for row in gen_data:
                row_copy = row.copy()
                row_copy['generation'] = gen_idx
                combined_data.append(row_copy)
    
    if not combined_data:
        return
    
    filename = f"{save_dir}/all_generations_data.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['generation', 'individual_id', 'throughput', 
                                               'cost', 'allocation', 'binding', 'total_pe_count'])
        writer.writeheader()
        writer.writerows(combined_data)
    
    print(f"✓ All generations combined data saved: {filename}")
    
def plot_throughput_cost_evolution(all_generations_data, save_dir="evolution_data"):
    """
    Create plots showing evolution of throughput and cost across generations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    generations = []
    best_throughputs = []
    best_costs = []
    avg_throughputs = []
    avg_costs = []
    
    for gen_idx, gen_data in enumerate(all_generations_data):
        if gen_data:
            generations.append(gen_idx)
            best_throughputs.append(max(d['throughput'] for d in gen_data))
            best_costs.append(min(d['cost'] for d in gen_data))
            avg_throughputs.append(np.mean([d['throughput'] for d in gen_data]))
            avg_costs.append(np.mean([d['cost'] for d in gen_data]))
    
    if not generations:
        return
    
    # Plot 1: Throughput evolution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_throughputs, 'b-', label='Best Throughput', linewidth=2, marker='o')
    plt.plot(generations, avg_throughputs, 'b--', label='Average Throughput', linewidth=1, alpha=0.7)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Throughput', fontsize=12)
    plt.title('Throughput Evolution Across Generations', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost evolution
    plt.subplot(1, 2, 2)
    plt.plot(generations, best_costs, 'r-', label='Best Cost', linewidth=2, marker='o')
    plt.plot(generations, avg_costs, 'r--', label='Average Cost', linewidth=1, alpha=0.7)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost Evolution Across Generations', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_dir}/evolution_plots.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Evolution plots saved: {filename}")

def save_pareto_front_data(front, generation, save_dir="evolution_data"):
    """
    Save Pareto front solutions to a separate file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    data = []
    for idx, ind in enumerate(front):
        throughput = ind.fitness.values[0]
        cost = ind.fitness.values[1]
        alloc = ind[:problem.n_types]
        binding = ind[problem.n_types:]
        
        data.append({
            'pareto_index': idx,
            'throughput': throughput,
            'cost': cost,
            'allocation': str(list(alloc)),   # Fixed: convert to list first
            'binding': str(list(binding)),     # Fixed: convert to list first
            'total_pe_count': sum(alloc)
        })
    
    # Sort by throughput descending
    data.sort(key=lambda x: x['throughput'], reverse=True)
    
    # Fix generation string for filename
    gen_str = str(generation) if isinstance(generation, str) else f"{generation:03d}"
    filename = f"{save_dir}/pareto_front_generation_{gen_str}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pareto_index', 'throughput', 'cost', 
                                               'allocation', 'binding', 'total_pe_count'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✓ Pareto front data saved: {filename}")
    
    return data


# ==============================
# Main Evolution Loop
# ==============================

def main():

    cleanup_old_xml_files()

    pop = toolbox.population(n=100)
    NGEN = 20
    MU = 50
    LAMBDA = 50
    CXPB = 0.9
    MUTPB = 0.7

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    save_evolution_plot(pop, 0)
    # operator new - save generation data
    all_generations_data = []
    gen_data = save_generation_data(pop, 0)
    all_generations_data.append(gen_data)

    for gen in range(1, NGEN + 1):
        
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        
        pop = toolbox.select(pop + offspring, MU)
        
        save_evolution_plot(pop, gen)
        # operator new - save generation data
        gen_data = save_generation_data(pop, gen)
        all_generations_data.append(gen_data)
        
        # operator new - save Pareto front every 5 generations (optional)
        if gen % 5 == 0 or gen == NGEN:
            front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
            save_pareto_front_data(front, gen)
        
        best_throughput = max(ind.fitness.values[0] for ind in pop if ind.fitness.valid)
        best_cost = min(ind.fitness.values[1] for ind in pop if ind.fitness.valid)
        print(f"Generation {gen}: Best Throughput={best_throughput:.4f}, Best Cost={best_cost:.2f}")

    # operator new - save combined data and plots
    save_all_generations_data(all_generations_data)
    plot_throughput_cost_evolution(all_generations_data)
    
    front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    # operator new - save final Pareto front
    save_pareto_front_data(front, "final")
    
    print("\n" + "="*50)
    print("Final Pareto Front:")
    print("="*50)
    for i, ind in enumerate(sorted(front, key=lambda x: x.fitness.values[0], reverse=True)):
        print(f"{i+1}. Throughput: {ind.fitness.values[0]:.6f} | Cost: {ind.fitness.values[1]:.2f}")

if __name__ == "__main__":
    main()