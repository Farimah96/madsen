import random
import numpy as np
import os
import xml.etree.ElementTree as ET
import subprocess
from xml.dom import minidom
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx


class Problem():

    def __init__(self):
        pass

    def appModel(self):

        # SADF model
        self.scenarios = ['s1', 's2', 's3']

        self.SADF_graph = [
            ["A", "B", 2, 'a'], # (src, dst, prod rate of src, cons rate of dst)
            ["B", "A", 'a', 2],
            ["B", "C", 'b', 'c'],
            ["C", "D", 'd', 'e'],
            ["D", "B", 1, 2],
            ["A", "E", 1, 1],
            ["E", "B", 'p', 1],
            ["E", "D", 'q', 1],
            ["E", "C", 'r', 1]
        ]

        self.initial_tokens = [('A', 'B', 0),
                        ('B', 'A', 10),
                        ('B', 'C', 0),
                        ('C', 'D', 0),
                        ('D', 'B', 3),
                        ('E', 'D', 0),
                        ('E', 'B', 0),
                        ('A', 'E', 0),
                        ('E', 'C', 0),]

        self.parametrised_rates = [
                                [2, 1, 1],
                                [3, 3, 2],
                                [1, 2, 1],
                                [2, 4, 1],
                                [3, 3, 1],
                                [1, 2, 2],
                                [3, 3, 4],
                                [2, 4, 4]]

        self.df = pd.DataFrame(self.parametrised_rates, index=["a", "b", "c", "d", "e", "p", "q", "r"],
                            columns = ["s1", "s2", "s3"]
                            )

        self.kernels = ['A', 'B', 'C', 'D']
        self.detectors = ['E']

        self.fsm = [('s1', 's2', 0.71),
                    ('s2', 's3', 1.00),
                    ('s3', 's2', 0.50),
                    ('s3', 's1', 0.20),
                    ('s1', 's3', 0.28),
                    ('s3', 's3', 0.30)
                    ]

        #######################################
        g_sadf = nx.DiGraph([(u, v, {'tokens': tok}) for u, v, tok in self.initial_tokens])
        g_fsm  = nx.DiGraph([(u, v, {'prob':   p  }) for u, v, p   in self.fsm])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        pos_sadf = nx.spring_layout(g_sadf, seed=7, k=1.6, iterations=200)
        pos_sadf['E'] = (pos_sadf['E'][0] * 1.6, pos_sadf['E'][1] * 1.6)

        nx.draw_networkx_nodes(g_sadf, pos_sadf, ax=ax,
                               node_color='lightblue', node_size=1600, edgecolors='black')
        nx.draw_networkx_labels(g_sadf, pos_sadf, ax=ax, font_weight='bold', font_size=11)

        normal_edges = [(u, v) for u, v, d in g_sadf.edges(data=True) if u != 'e' and v != 'e']
        dashed_edges = [(u, v) for u, v, d in g_sadf.edges(data=True) if u == 'e' or v == 'e']

        nx.draw_networkx_edges(g_sadf, pos_sadf, ax=ax, edgelist=normal_edges,
                               arrows=True, arrowsize=18, width=1.6,
                               connectionstyle='arc3,rad=0.08',
                               min_target_margin=18, min_source_margin=18)
        nx.draw_networkx_edges(g_sadf, pos_sadf, ax=ax, edgelist=dashed_edges,
                               arrows=True, arrowsize=18, width=1.6, style='dashed',
                               connectionstyle='arc3,rad=0.08',
                               min_target_margin=18, min_source_margin=18)

        edge_labels = {(u, v): d['tokens'] for u, v, d in g_sadf.edges(data=True)}
        nx.draw_networkx_edge_labels(g_sadf, pos_sadf, edge_labels=edge_labels, ax=ax,
                                     font_size=9, label_pos=0.35,
                                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        ax.set_title("SADF Application Graph")
        ax.axis('off')

        ax = axes[1]
        pos_fsm = {'s1': (0.0, 1.0), 's2': (1.0, 1.0), 's3': (0.5, 0.0)}

        nx.draw_networkx_nodes(g_fsm, pos_fsm, ax=ax,
                               node_color='lightgreen', node_size=1800, edgecolors='black')
        nx.draw_networkx_labels(g_fsm, pos_fsm, ax=ax, font_weight='bold', font_size=12)

        selfloops    = [(u, v) for u, v, d in g_fsm.edges(data=True) if u == v]
        normal_fsm   = [(u, v) for u, v, d in g_fsm.edges(data=True) if u != v]

        rad_map = {('s1', 's2'):  0.12,
                   ('s2', 's3'): -0.12,
                   ('s3', 's2'): -0.35,
                   ('s3', 's1'):  0.35,
                   ('s1', 's3'):  0.12}

        for (u, v) in normal_fsm:
            nx.draw_networkx_edges(g_fsm, pos_fsm, ax=ax, edgelist=[(u, v)],
                                   arrows=True, arrowsize=18, width=1.6,
                                   connectionstyle=f'arc3,rad={rad_map.get((u, v), 0.15)}',
                                   min_target_margin=22, min_source_margin=22)

        for (u, v) in selfloops:
            nx.draw_networkx_edges(g_fsm, pos_fsm, ax=ax, edgelist=[(u, v)],
                                   arrows=True, arrowsize=18, width=1.6,
                                   connectionstyle='arc3,rad=0.6',
                                   min_target_margin=22, min_source_margin=22)

        fsm_labels = {(u, v): f"{d['prob']:.2f}" for u, v, d in g_fsm.edges(data=True)}
        nx.draw_networkx_edge_labels(g_fsm, pos_fsm, edge_labels=fsm_labels, ax=ax,
                                     font_size=10,
                                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

        if ('s3', 's3') in g_fsm.edges:
            x, y = pos_fsm['s3']
            ax.annotate('0.30', xy=(x + 0.18, y - 0.28), fontsize=10,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.95))

        ax.set_title("FSM Scenario Transition Graph")
        ax.axis('off')

        ax.margins(0.25)
        axes[0].margins(0.25)

        plt.tight_layout()
        plt.savefig("graphs.png", dpi=150, bbox_inches='tight')
        plt.show()

    def create_random_architecture(
            num_tiles=4,
            processor_types=["cpu", "gpu", "gcc", "fpga"],
            mem_range=(2000, 8000),
            bw_range=(8.0, 20.0),
            max_ni_range=(4, 16),
            latency_range=(1.0, 5.0),
            connectivity_prob=0.8
        ):

        tiles = {}

        for i in range(1, num_tiles + 1):
            tile_id = f"t{i}"
            tiles[tile_id] = {
                "pt": random.choice(processor_types),
                "mem": random.randint(mem_range[0], mem_range[1]),
                "max_ni_conns": random.randint(max_ni_range[0], max_ni_range[1]),
                "bw": round(random.uniform(bw_range[0], bw_range[1]), 1)
            }

        connections = []
        tile_ids = list(tiles.keys())

        for u in tile_ids:
            for v in tile_ids:
                if u != v:
                    if random.random() < connectivity_prob:
                        connections.append({
                            "src": u,
                            "dst": v,
                            "latency": round(random.uniform(latency_range[0], latency_range[1]), 1)
                        })

        return {"tiles": tiles, "connections": connections}

    def decode_chromosome(chromosome):
        processor_types=["cpu", "gpu", "gcc", "fpga"]
        allocation = chromosome[:5]
        expanded_allocation = []
        for gene in allocation:
            for n in range(gene):
                expanded_allocation.append(processor_types[gene])
        binding = chromosome[5:]

        processor_count = 0
        for gene in allocation:
            processor_count = gene + processor_count

        return expanded_allocation, binding, processor_count

    def mapping(binding, architecture, expanded_allocation):

        for actor in range(len(binding)):
            processor_index = binding[actor]
            if processor_index < len(expanded_allocation):
                processor_type = expanded_allocation[processor_index]
                for tile_id, tile_info in architecture['tiles'].items():
                    if tile_info['pt'] == processor_type:
                        print(f"Task {actor} is mapped to Tile {tile_id} with Processor Type {processor_type}")
                        break
            else:
                print(f"Task {actor} has an invalid binding index: {processor_index}")



    def get_next_scenarios(fsm, current_scenario):
        possible_next_scenarios = [v for u, v, p in fsm if u == current_scenario]  # check: extracted transitions properly
        if not possible_next_scenarios:
            return None

        probabilities = [p for u, v, p in fsm if u == current_scenario]  # check: aligned weights with transitions
        next_scenario = random.choices(possible_next_scenarios, weights=probabilities, k=1)[0]
        return next_scenario

    def inject_scenario( df, SADF_graph, kernels, next_scenario):
        actors_cons_rates = []
        actors_prod_rates = []
        if next_scenario is None:  # check: guard against terminal scenarios
            return None, [], []

        for actor in kernels:
            # if actor exists in the SADF graph lists, based on src or dst, get the corresponding rates from the SADF graph lists that relates to each edge
            # if its rate was a number, take that number for prod or cons rate else take the related number of that parameter from df
            # if actor location in SADF_graph lists was [0], production rate of it equals to [2] in that list
            # if actor location in SADF_graph lists was [1], consumption rate of it equals to [3] in that list
            if actor in [edge[0] for edge in SADF_graph]:  #actor is a source
                edge = next(edge for edge in SADF_graph if edge[0] == actor)
                prod_rate = edge[2]
                if isinstance(prod_rate, str):  # if it's a parameter, look it up in df
                    prod_rate = df.loc[prod_rate, next_scenario]
                actors_prod_rates.append(prod_rate)
            elif actor in [edge[1] for edge in SADF_graph]:  # actor is a destination
                edge = next(edge for edge in SADF_graph if edge[1] == actor)
                cons_rate = edge[3]
                if isinstance(cons_rate, str):  #if it's a parameter, look it up in df
                    cons_rate = df.loc[cons_rate, next_scenario]
                actors_cons_rates.append(cons_rate)
            print(f"Actor {actor}: Production Rate = {actors_prod_rates[-1] if actors_prod_rates else 'N/A'}, Consumption Rate = {actors_cons_rates[-1] if actors_cons_rates else 'N/A'}")
        return next_scenario, actors_cons_rates, actors_prod_rates 



if __name__ == "__main__":
    problem = Problem()
    problem.appModel()

    random_arch = Problem.create_random_architecture(
        num_tiles=4,
        processor_types=["cpu", "gpu", "gcc", "fpga"],
        mem_range=(2000, 8000),
        bw_range=(8.0, 20.0),
        max_ni_range=(4, 16),
        latency_range=(1.0, 5.0),
        connectivity_prob=0.8
    )

    chromosome = [1, 0, 2, 1, 0, 0, 1, 2, 0, 1]
    expanded_allocation, binding, processor_count = Problem.decode_chromosome(chromosome)
    Problem.mapping(binding, random_arch, expanded_allocation)

    print(f"Total Tiles: {len(random_arch['tiles'])}")
    for tile_id, tile_info in random_arch['tiles'].items():
        print(f"Tile {tile_id}: {tile_info}")
    print(f"Total Interconnect Links: {len(random_arch['connections'])}")
    for conn in random_arch['connections']:
        print(f"Link from {conn['src']} to {conn['dst']} with latency {conn['latency']}")

    current_scenario = 's1'
    detector_result = Problem.get_next_scenarios(problem.fsm, current_scenario)
    print(f"Current Scenario: {current_scenario}, Next Scenario: {detector_result}")
    cons_rates, prod_rates = [], []
    next_scenario, cons_rates, prod_rates = Problem.inject_scenario(problem.df, problem.SADF_graph, problem.kernels, detector_result)
    