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

class Representation(): # encode, evaluate, 
    def __init__(self, tasks, weights, resource_graph, exec_time, communication_time, pe_types):
        self.tasks = tasks
        self.weights = weights
        self.resource_graph = resource_graph
        self.exec_time = exec_time
        self.communication_time = communication_time
        self.pe_types = pe_types
        
    def encode(chromosome, tasks): #return dictionary of task:pe
        mapping = {}
        for i, task in enumerate(tasks):
            mapping[task] = chromosome[i]
        print("mapping is:", mapping)
        return mapping
    
    # def evaluate(chromosome): # time of task exec must be valid an
        
    # def repair(chromosome): # if chromosome is not valid -> repair



class fixed_arch_problem(ElementwiseProblem):
    def __init__(self, elementwise=True, **kwargs):
        super().__init__(elementwise,n_var=5, n_obj=3, **kwargs)

        
    
    def static_list_scheduler(tasks, weights, resource_graph, exec_time, communication_time, pe_types):
        
        
        def return_parents(node):
            parents = []
            for edge in weights.keys():
                if edge[1] == node:
                    parents.append(edge[0])
            return parents
        
        
        def return_children(node):
            children = []
            for edge in weights.keys():
                if edge[0] == node:
                    children.append(edge[1])
            return children
        
        
        # def add_dummy_node(tasks, weights, exec_time, resource_graph):
            
            # finding roots
            
            roots = []
            allOfChilds = []
            for task in tasks:
                childs = return_children(task)
                allOfChilds.extend(childs)
            for task in tasks:
                if task not in allOfChilds:
                    roots.append(task)
                    
                    
            # finding leaves        
            leaves = []
            allOfParents = []
            for task in tasks:
                parents = return_parents(task)
                allOfParents.extend(parents)
            for task in tasks:
                if task not in allOfParents:
                    leaves.append(task)
                           
            # adding main root
            for root in roots:
                weights["mainRoot", root] = 0
            print("weights after adding main root:", weights)
            
            print("\n") 
            
            #adding final leaf
            for leaf in leaves:
                weights[leaf, "finalRoot"] = 0
            print("weights after adding final leaf:", weights)
            
            # add new nodes to tasks
            if "mainRoot" not in tasks:
                tasks.append("mainRoot")
                exec_time.append([0]*len(exec_time[0]))
                resource_graph["mainRoot"] = "fpga"
            
            if "finalRoot" not in tasks:
                tasks.append("finalRoot")
                exec_time.append([0]*len(exec_time[0]))
                resource_graph["finalRoot"] = "fpga"
            
            print("\n")
            
            print("tasks after adding dummy nodes:", tasks)           
                    
            print("leaves and roots are: ", leaves, roots)
            return leaves, roots
                
        
        def topologyList(tasks):
            in_degree = {task: 0 for task in tasks}
            for edge in weights.keys():
                in_degree[edge[1]] += 1

            zero_in_degree = [task for task in tasks if in_degree[task] == 0]
            topo_order = []

            while zero_in_degree:
                current_task = zero_in_degree.pop(0)
                topo_order.append(current_task)

                for child in return_children(current_task):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        zero_in_degree.append(child)


            return topo_order
        
        
        def reverse_topologyList(tasks):
            out_degree = {task: 0 for task in tasks}
            for edge in weights.keys():
                out_degree[edge[0]] += 1

            zero_out_degree = [task for task in tasks if out_degree[task] == 0]
            rev_topo_order = []

            while zero_out_degree:
                current_task = zero_out_degree.pop(0)
                rev_topo_order.append(current_task)

                for parent in return_parents(current_task):
                    out_degree[parent] -= 1
                    if out_degree[parent] == 0:
                        zero_out_degree.append(parent)

            return rev_topo_order
        
        

        def compute_t_level(TopList, pe_types):
            t_level = {task : 0 for task in TopList}
            
            for node in TopList:
                max = 0
                parents = return_parents(node)
                if not parents:
                    t_level[node] = 0
                    continue
                for parent in parents:
                    if parent in tasks:  # to avoid dummy nodes -> look at 9 lines above
                        if t_level[parent] + weights[(parent, node)] + exec_time[tasks.index(parent)][pe_types.index(resource_graph[parent])] > max:
                            max = t_level[parent] + weights[(parent, node)] + exec_time[tasks.index(parent)][pe_types.index(resource_graph[parent])]
                
                    t_level[node] = max

            
            for node in TopList:
                print("t_level of node", node, "is", t_level[node])
                    
            print(t_level)
                    
            print("\n")
            return t_level  # return dict 

    

        def compute_b_level(RevTopList, pe_types):
            b_level = {task : 0 for task in RevTopList}
            for node in RevTopList:
                max = 0
                childrens = return_children(node)
                if not childrens:
                    b_level[node] = 0
                    continue
                for child in childrens:
                    if child in tasks:  # to avoid dummy nodes -> look at 9 lines above
                        if b_level[child] + weights[(node, child)]+ exec_time[tasks.index(child)][pe_types.index(resource_graph[child])] > max:
                            max = b_level[child] + weights[(node, child)] + exec_time[tasks.index(child)][pe_types.index(resource_graph[child])]
                    b_level[node] = max
                
            for node in RevTopList:
                print("b_level of node", node, "is", b_level[node])
                
            b_level_rev = dict(reversed(list(b_level.items())))
            print(b_level_rev)
            print("\n")
            return b_level_rev  # return dict
            

        
        def Computing_priorities(nodes, t_level, b_level):
            priority_list = {}
            for node in nodes:
                if node == "mainRoot" or node == "finalRoot":
                    continue
                priority = b_level[node] + t_level[node]
                priority_list.update({node:priority})
                print("priority of node", node, "is", b_level[node] + t_level[node])
            sorted_priority_list = dict(sorted(priority_list.items(), key=lambda item: item[1], reverse=True))
            return sorted_priority_list


            
        def NumـUnschedueldـPredecessors(tasks, weights):
                prec_dict = {task: 0 for task in tasks}
                prec_task_names = {task: [] for task in tasks}
                unscheduled_tasks = list(tasks)
                
                for edge in weights.keys():
                    if edge[1] in prec_dict:  # to avoid dummy nodes -> look at 9 lines above
                        prec_dict[edge[1]] += 1
                        prec_task_names[edge[1]].append(edge[0])

                return unscheduled_tasks
                        

        
        def find_est(task, pe, scheduled_tasks):
            if not scheduled_tasks:
                return 0
            else:
                max = 0
                parents = return_parents(task)
                for parent in parents:
                    if parent in scheduled_tasks.keys():
                        parent_pe = scheduled_tasks[parent][1] #exec time and which pe
                        print("parent", parent, "is scheduled on pe", parent_pe)
                        # there is no communication time yet
                        parent_end_time = scheduled_tasks[parent][0] + exec_time[tasks.index(parent)][parent_pe]  # 1 -> 0
                        print("parent", parent, "ends at", parent_end_time)
                        if parent_pe != pe:     #parent is scheduled on a different pe
                            parent_end_time += communication_time
                        if parent_end_time > max:
                            max = parent_end_time
                return max
            




        def condition_passed(task, scheduled_tasks, est):
            # get only real parents and ignore any dummy nodes that aren't in tasks
            parents = [p for p in return_parents(task) if p in tasks]

            # if task has no real parents -> it's a root
            if not parents:
                return True

            # otherwise, it's ready only if all parents have been scheduled
            for parent in parents:
                if parent not in scheduled_tasks:
                    return False
            
            pe = resource_graph[task]
            running_on_pe = [
                t for t, v in scheduled_tasks.items()
                if pe_types[v[1]] == pe
                and (v[0] + exec_time[tasks.index(t)][v[1]]) > est
            ]
            if len(running_on_pe) >= pe_types.index(pe):
                return False
            
            return True
  

          
        print("scheduling with static list scheduling algorithm:\n")  

        
        
        def final_scheduler(tasks, weights, resource_graph, pe_types):            
            scheduled_tasks = {} # task : (start_time, pe)
            # add_dummy_node(tasks, weights, exec_time, resource_graph)
            unscheduled_tasks = NumـUnschedueldـPredecessors(tasks, weights)
            print("initially unscheduled tasks are:", unscheduled_tasks)

            TopList = topologyList(tasks)
            print("Topological List:", TopList)  
            RevTopList = reverse_topologyList(tasks)
            print("Reverse Topological List:", RevTopList)     
            print("\n")

            t_level = compute_t_level(topologyList(tasks), pe_types)
            b_level = compute_b_level(reverse_topologyList(tasks), pe_types)
            priority_list = Computing_priorities(tasks, t_level, b_level)
            
            print("initially priority list is:", priority_list)
            
            timeline = []  # list of (task, pe_type, start, end)
            
            while unscheduled_tasks:
                # if unscheduled_tasks == ["mainRoot","finalRoot"]:
                #     break
                for task in priority_list.keys():
                    if task in unscheduled_tasks and condition_passed(task, scheduled_tasks, find_est(task, pe_types.index(resource_graph[task]), scheduled_tasks)):
                        y = task
                        print("\n")
                        print("y is set to", y)
                        break
                
                pe = resource_graph.get(y)
                pe_index = pe_types.index(pe) if pe in pe_types else 0
                est = find_est(y, pe_index, scheduled_tasks)
                

                scheduled_tasks[y] = (est, pe_index) # (start_time, pe)
                print("task", y, "is scheduled at time", est, "on pe", pe)

                unscheduled_tasks.remove(y)
                
                print("unscheduled tasks after removing", y, "are:", unscheduled_tasks)
                print("scheduled tasks so far are:", scheduled_tasks)             
                print("\n")
                  
   
                
            print("final scheduled tasks are:", scheduled_tasks)
            return scheduled_tasks
                
                        
        final_scheduler(tasks, weights, resource_graph, pe_types)
        
        
        
###############################################################################################################
###                                           SAMPLING CLASS                                                ###
###############################################################################################################      
        


class MySampling(Sampling):

    def __init__(self, num_pes=3, fixed_indices=None, fixed_values=None): 
        super().__init__()
        self.num_pes = int(num_pes)
        self.fixed_indices = fixed_indices if fixed_indices is not None else []
        self.fixed_values = fixed_values if fixed_values is not None else []

    def _do(self, problem, n_samples, **kwargs):
        samples = np.random.randint(0, self.num_pes, size=(n_samples, problem.n_var))

        if self.fixed_indices:
            for idx, val in zip(self.fixed_indices, self.fixed_values):
                samples[:, idx] = val

        return samples
    
