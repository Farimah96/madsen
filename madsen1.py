""" Implementation of mapping problem with fix platform. """

import string

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
import networkx as nx

# Application Model
""" 1 task graph for simple mode """



class fixed_arch_problem(ElementwiseProblem):
    def __init__(self, elementwise=True, **kwargs):
        super().__init__(elementwise,n_var=4, n_obj=3, **kwargs)
        
        global tasks
        tasks = ["a", "b", "c", "d"]
        
        global exec_time
        exec_time = [
            [0.9, 1.4, 0.7, 1.4],  # task 0 on PEs 0..3
            [1.1, 1.0, 0.6, 1.0],
            [0.8, 1.2, 0.9, 1.2],
            [1.3, 0.9, 0.7, 0.9]
        ]
        
        global weights
        weights = {("a" , "b"):2, ("a" , "c"):4, ("b" , "c"):5, ("b" , "d"):3, ("c" , "d"):1}
        
        global pe_types
        pe_types = ["fpga", "asic", "gpp"] #0 2 1 2        
        
        global fix_pe
        fix_pe = {"a" : pe_types[0], "b" : pe_types[3], "c" :pe_types[1], "d" : pe_types[2]}        
        
        global communication_time
        communication_time = 5        
        
        global deadlines
        deadlines = [30, 20, 40, 60]        
        
        
        ########################### useless for now ###########################
        
        self.n_tasks = 4
        self.n_edges = 5
        
        e1 = 2
        e2 = 4
        e3 = 1
        e4 = 7
        e5 = 3
        
        
        self.edges = [
            (1,2,e1),
            (1,3,e2),
            (2,3,e3),
            (2,4,e4),
            (3,4,e5)
        ]
        
        self.pe_power = [2.2, 1.2, 1.8, 1.2]  # watts
        

        

        

        periods = [30, 20, 20, 30]
        # execution time, ci, the consumed energy, ei, and the memory usage, mi, are determined by the actual mapping.
        
        
        pe_n = 4

        bus_n = 2
        bridge_ = 1
        #ignore bus BW and bus power consumption
        
        # fixed architecture

        
        buses = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        bridges = [1, 2]
        
 

        
    
    def static_list_scheduler(tasks, weights, exec_time, communication_time):
        # Create a directed graph (https://stackoverflow.com/questions/48103119/drawing-a-network-with-nodes-and-edges-in-python3)

        # #draw the graph
        # G = nx.Graph()
        # # each edge is a tuple of the form (node1, node2, {'weight': weight})
        # edges = [(k[0], k[1], {'weight': v}) for k, v in weights.items()]
        # G.add_edges_from(edges)
        # pos = nx.spring_layout(G) # positions for all nodes
        # # nodes
        # nx.draw_networkx_nodes(G,pos,node_size=700)
        # # labels
        # nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
        # # edges
        # nx.draw_networkx_edges(G,pos,edgelist=edges, width=6)
        # # weights
        # labels = nx.get_edge_attributes(G,'weight')
        # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        # plt.show()

        #Computing a t-level
        TopList = ["a", "b", "c", "d"]
        def return_parents(node):
            parents = []
            for edge in weights.keys():
                if edge[1] == node:
                    parents.append(edge[0])
            return parents

        

        def compute_t_level(TopList):
            t_level = {"a":0, "b":0, "c":0, "d":0}
            for node in TopList:
                max = 0
                parents = return_parents(node)
                if not parents:
                    t_level[node] = 0
                    continue
                for parent in parents:
                    if t_level[parent] + weights[(parent, node)] > max:
                        max = t_level[parent] + weights[(parent, node)]
                
                t_level[node] = max

            return t_level  # return dict 
            # for node in TopList:
            #     print("t_level of node", node, "is", t_level[node])
                
            # print(t_level_list)
                
            # print("\n")
            
            
            
        #Computing a b-level

        RevTopList = ["d", "c", "b", "a"]

        def return_children(node):
            children = []
            for edge in weights.keys():
                if edge[0] == node:
                    children.append(edge[1])
            return children

        

        def compute_b_level(RevTopList):
            b_level = {"a":0, "b":0, "c":0, "d":0}
            for node in RevTopList:
                max = 0
                childrens = return_children(node)
                if not childrens:
                    b_level[node] = 0
                    continue
                for child in childrens:
                    if b_level[child] + weights[(node, child)] > max:
                        max = b_level[child] + weights[(node, child)]
                b_level[node] = max
                
            # for node in RevTopList:
            #     print("b_level of node", node, "is", b_level[node])
            b_level_rev = dict(reversed(list(b_level.items())))
            return b_level_rev  # return dict
            # print(b_level_list_rev)
                
            # print("\n")

        
        def Computing_priorities(nodes, t_level, b_level):
            priority_list = {}
            minpriority = -1000
            for node in nodes:
                if b_level[node] + t_level[node] > minpriority:
                    minpriority = b_level[node] + t_level[node]
                    priority_list.update({node:minpriority})
                print("priority of node", node, "is", b_level[node] + t_level[node])
                
            print(priority_list)
            print("\n")
                    
        t_level = compute_t_level(TopList)                                             ###########################
        b_level = compute_b_level(RevTopList)                                          #########   Test   ########
        Computing_priorities(tasks, t_level, b_level)                                  ########################### 

        def NumـUnschedueldـPredecessors():
            unscheduled_tasks = []
            prec = {}
            prec_task_name = {}
            for task in tasks:
                if task == tasks[0]:
                    prec[task] = 0
                else:
                    prec[task] = 0
                    for edge in weights.keys():
                        if edge[1] == task:
                            prec[task] += 1
                            prec_task_name[task] = edge[0]                    
                            
                unscheduled_tasks.append(task)
            # print("Number of unscheduled predecessors for each task:", prec)
            # print("The predecessor task name for each task:", prec_task_name)
            return prec, unscheduled_tasks
        
        # NumـUnschedueldـPredecessors()
        
        def condition_passed(task, scheduled_tasks):  # not tested yet and add condition 2
            parents = return_parents(task)
            for parent in parents:
                if parent not in scheduled_tasks:
                    return False
            return True
            
        def find_est(task, pe, scheduled_tasks):
            if not scheduled_tasks:
                return 0
            else:
                max = 0
                parents = return_parents(task)
                for parent in parents:
                    if parent in scheduled_tasks.keys(): #scheduled_tasks is a dict of scheduled tasks with (start_time, pe, end_time)
                        parent_pe = scheduled_tasks[parent][1]
                        print("parent", parent, "is scheduled on pe", parent_pe)
                        parent_end_time = scheduled_tasks[parent][2] + exec_time[tasks.index(parent)][parent_pe]
                        print("parent", parent, "ends at", parent_end_time)
                        if parent_pe != pe:     #parent is scheduled on a different pe
                            parent_end_time += communication_time
                        if parent_end_time > max:
                            max = parent_end_time
                print("EST of task", task, "on pe", pe, "is", max)
                return max
            
        # task = "c"
        # pe = "fpga"
        # scheduled_tasks = {"b": (0, 1), "a": (1.0, 0)} 
        # find_est(task, pe, scheduled_tasks)