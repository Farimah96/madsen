""" Implementation of mapping problem with fix platform. """
import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling


from pymoo.util.ref_dirs import get_reference_directions


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

        # each row means a bus and each column means a PE
        buses = [
            [1,1,0,0,0]
            [0,0,1,1,1]
        ]
        
        # each row means a PE and each column means a bus
        bridges = [1,2]
    
        
    def encode(chromosome, tasks): #return dictionary of task:pe
        mapping = {}
        for i, task in enumerate(tasks):
            mapping[task] = chromosome[i]
        print("mapping is:", mapping)
        return mapping
    

class fixed_arch_problem(ElementwiseProblem):
    def __init__(self, elementwise=True, **kwargs):
        super().__init__(elementwise,n_var=5, n_obj=2, **kwargs)

        self.tasks = ["a", "b", "c", "d", "x"]
        
        self.resource_graph = {"a" : "fpga", "b" : "gpp", "c" : "asic", "d" : "gpp", "x" : "fpga"}
        
        self.weights = {
            ("a", "b"): 2,
            ("x", "b"): 3,
            ("a", "c"): 4,
            ("b", "c"): 5,
            ("b", "d"): 3,
            ("c", "d"): 1
        }
        
        self.exec_time = [
            [0.9, 1.4, 0.7, 1.0],  # task a
            [1.1, 1.0, 0.6, 0.9],  # task b
            [0.8, 1.2, 0.9, 1.1],  # task c
            [1.3, 0.9, 0.7, 1.2],  # task d
            [0, 0, 0, 0],          # task x
        ]
        
        self.pe_types = ["fpga", "gpp", "asic", "gpp"]
        
        self.communication_time = 5
    
    def static_list_scheduler(self, tasks, weights, resource_graph, exec_time, communication_time, pe_types):
        
        
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

            
            # for node in TopList:
                # print("t_level of node", node, "is", t_level[node])
                    
            # print(t_level)
                    
            # print("\n")
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
                
            # for node in RevTopList:
                # print("b_level of node", node, "is", b_level[node])
                
            b_level_rev = dict(reversed(list(b_level.items())))
            # print(b_level_rev)
            # print("\n")
            return b_level_rev  # return dict
            

        
        def Computing_priorities(nodes, t_level, b_level):
            priority_list = {}
            for node in nodes:
                if node == "mainRoot" or node == "finalRoot":
                    continue
                priority = b_level[node] + t_level[node]
                priority_list.update({node:priority})
                # print("priority of node", node, "is", b_level[node] + t_level[node])
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
                        

        #find earliest start time just based on execution time of parents and 
        #ignore communication time
        def find_est(task, pe, scheduled_tasks):
            if not scheduled_tasks:
                return 0
            else:
                max = 0
                parents = return_parents(task)
                for parent in parents:
                    if parent in scheduled_tasks.keys():
                        parent_pe = scheduled_tasks[parent][1] #exec time and which pe
                        parent_end_time = scheduled_tasks[parent][0] + exec_time[tasks.index(parent)][parent_pe]  # 1 -> 0
                        if parent_end_time > max:
                            max = parent_end_time
                        else:
                            continue
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
            
            # pe = resource_graph[task]
            # running_on_pe = [
            #     t for t, v in scheduled_tasks.items()
            #     if pe_types[v[1]] == pe
            #     and (v[0] + exec_time[tasks.index(t)][v[1]]) > est
            # ]
            # if len(running_on_pe) >= pe_types.index(pe):
            #     return False
            
            return True
  

          
        print("scheduling with static list scheduling algorithm:\n")  

        
        
        def final_scheduler(tasks, weights, resource_graph, pe_types):            
            scheduled_tasks = {} # task : (start_time, pe)
            # add_dummy_node(tasks, weights, exec_time, resource_graph)
            unscheduled_tasks = NumـUnschedueldـPredecessors(tasks, weights)
            # print("initially unscheduled tasks are:", unscheduled_tasks)

            TopList = topologyList(tasks)
            # print("Topological List:", TopList)  
            RevTopList = reverse_topologyList(tasks)
            # print("Reverse Topological List:", RevTopList)     
            # print("\n")

            t_level = compute_t_level(topologyList(tasks), pe_types)
            b_level = compute_b_level(reverse_topologyList(tasks), pe_types)
            priority_list = Computing_priorities(tasks, t_level, b_level)
            
            # print("initially priority list is:", priority_list)
            
            timeline = []  # list of (task, pe_type, start, end)
            
            while unscheduled_tasks:
                # if unscheduled_tasks == ["mainRoot","finalRoot"]:
                #     break
                for task in priority_list.keys():
                    if task in unscheduled_tasks and condition_passed(task, scheduled_tasks, find_est(task, pe_types.index(resource_graph[task]), scheduled_tasks)):
                        y = task
                        # print("\n")
                        # print("y is set to", y)
                        break
                
                pe = resource_graph.get(y)
                pe_index = pe_types.index(pe) if pe in pe_types else 0
                est = find_est(y, pe_index, scheduled_tasks)
                

                scheduled_tasks[y] = (est, pe_index) # (start_time, pe)
                # print("task", y, "is scheduled at time", est, "on pe", pe)

                unscheduled_tasks.remove(y)
                
                # print("unscheduled tasks after removing", y, "are:", unscheduled_tasks)
                # print("scheduled tasks so far are:", scheduled_tasks)             
                # print("\n")
                  
   
                
            print("final scheduled tasks are:", scheduled_tasks)
            
            #for each task print details of scheduling including start time and executing time of 
            # pe and communication time from weights dictionary
            for task, (start, pe) in scheduled_tasks.items():
                exec_t = exec_time[tasks.index(task)][pe]
                print(f"Task {task} is scheduled on PE {pe_types[pe]} starting at time {start} with execution time {exec_t}")
                parents = return_parents(task)
                for parent in parents:
                    if parent in scheduled_tasks:
                        comm_t = weights.get((parent, task), 0)
                        if comm_t > 0:
                            print(f"  Communication time from parent task {parent} to task {task} is {comm_t}")
                
            return scheduled_tasks
                
                        
        return final_scheduler(tasks, weights, resource_graph, pe_types)
    
        
    def _evaluate(self, x, out, *args, **kwargs):
        mapping = {self.tasks[i]: self.pe_types[int(x[i] % len(self.pe_types))] for i in range(len(self.tasks))}
        sch_dict = self.static_list_scheduler(tasks=self.tasks,
                                      weights=self.weights,
                                      resource_graph=mapping,
                                      exec_time=self.exec_time,
                                      communication_time=self.communication_time,
                                      pe_types=self.pe_types)
        
        
        system_cost = {"fpga": 100, "gpp": 50, "asic": 150}
        cost = sum(system_cost[mapping[t]] for t in mapping)
        
        
        finish_time = 0
        for task_name, info in sch_dict.items():
            if info[0] + self.exec_time[self.tasks.index(task_name)][info[1]] > finish_time:
                finish_time = info[0] + self.exec_time[self.tasks.index(task_name)][info[1]]
                
        print("Makespan is:", finish_time)
        print("Cost is:", cost)
        out["F"] = np.array([finish_time, cost])
        
        
        
        
        
        
###############################################################################################################
###                                           SAMPLING & ...                                                ###
###############################################################################################################      
        


class MySampling(Sampling):

    def __init__(self, num_pes=4, fixed_indices=None, fixed_values=None): 
        super().__init__()
        self.num_pes = int(num_pes)
        self.fixed_indices = fixed_indices if fixed_indices is not None else []
        self.fixed_values = fixed_values if fixed_values is not None else []
    
    def valid_sample(sample, resource_graph, pe_types): # sample is a list of pe indices
        for pe in sample:
            if pe < 0 or pe >= len(pe_types):
                return False
            if pe_types[pe] != resource_graph[pe]:
                return False
            if sample.count(pe) > pe_types.count(resource_graph[pe]):
                return False
        
        
    
    def _do(self, problem, n_samples, **kwargs):
        samples = []
        while len(samples) < n_samples:
            sample = []
            for i in range(problem.n_var):
                if i in self.fixed_indices:
                    index = self.fixed_indices.index(i)
                    sample.append(self.fixed_values[index])
                else:
                    pe_index = np.random.randint(0, self.num_pes)
                    sample.append(pe_index)
            samples.append(sample)
            print(sample)
        return np.array(samples)
    

class MyMutation(Mutation):
    def __init__(self, mutation_rate=0.4, num_pes=4):
        super().__init__()
        self.mutation_rate = mutation_rate
        self.num_pes = int(num_pes)
        
                # each row means a bus and each column means a PE
        buses = [
            [1,1,0,0,0],
            [0,0,1,1,1]
        ]
        
        # each row means a PE and each column means a bus
        bridges = [1,2]


        exec_time = [
            [0.9, 1.4, 0.7, 1.0],  # task a
            [1.1, 1.0, 0.6, 0.9],  # task b
            [0.8, 1.2, 0.9, 1.1],  # task c
            [1.3, 0.9, 0.7, 1.2],  # task d
            [0, 0, 0, 0],          # task x
        ]

        #Randomly select an existing PE and change its type,and randomly select a bus and change its type
        def _do(self, problem, X, **kwargs):
            Y = np.full_like(X, np.nan)
            for i in range(X.shape[0]):
                individual = X[i].copy()
                for j in range(len(individual)):
                    if np.random.rand() < self.mutation_rate:
                        new_pe = np.random.randint(0, self.num_pes)
                        individual[j] = new_pe
                Y[i] = individual
                print("individual after mutation: ", Y)
            return Y
            
            
            
        # def addPE(tasks, pe_types): #Add a new PE to a randomly selected bus, and assign 
        #     #Ceiling of ( VT / VPE ) tasks randomly selected from the other PEs
        #     random_bus = np.random.randint(0, len(buses))
        #     buses[random_bus].append(1)  # Add a new PE to the selected bus
        #     num_tasks_to_assign = int(np.ceil(len(tasks) / len(pe_types)))
        #     tasks_to_assign = np.random.choice(tasks, size=num_tasks_to_assign, replace=False)
        #     for task in tasks_to_assign:
        #         # Assign the task to the new PE
        #         pe_index = len(buses[random_bus]) - 1
        #         #update the task assignment logic
        #         ##################################
        #         ##################################
        #         ###################################
                
        
            
        # def removePE(): #Remove a PE from a randomly selected bus, and distribute its tasks 
        #     #among the remaining PEs
        #     random_bus = np.random.randint(0, len(buses))
        #     if len(buses[random_bus]) > 1:
        #         pe_index_to_remove = np.random.randint(0, len(buses[random_bus]))
        #         buses[random_bus].pop(pe_index_to_remove)
        #         #logic to redistribute tasks assigned to the removed PE
        #         ##################################
        #         ##################################
        #         ##################################
            
            
            
            
        # def ranReasTask(pe_types): #Move [1;4] randomly selected tasks from a PE to another 
        #     #randomly chosen PE
        #     seleced_pe_from = np.random.randint(0, len(pe_types))
        #     selected_pe_to = np.random.randint(0, len(pe_types))
        #     while selected_pe_to == seleced_pe_from:
        #         selected_pe_to = np.random.randint(0, len(pe_types))
        #     num_tasks_to_move = np.random.randint(1, 5)
        #     # kogic to move tasks from seleced_pe_from to selected_pe_to
        #     ################################################
        #     ################################################
        #     ################################################
            
            
            
            
        # def ranReasTask2(pe_types): #Move [1;4] randomly selected tasks from a PE to another 
        #     #PE that is connected to the same bus
        #     seleced_pe_from = np.random.randint(0, len(pe_types))
        #     # Logic to find PEs connected to the same bus
        #     connected_pes = []  # Placeholder for connected PEs
        #     if connected_pes:
        #         selected_pe_to = np.random.choice(connected_pes)
        #         num_tasks_to_move = np.random.randint(1, 5)
        #         # logic to move tasks from seleced_pe_from to selected_pe_to
        #         ##################################
        #         ##################################
        #         ##################################
                


        # def ranHeuReasTask(tasks, taskDeadlines, schedulesTasks): # taskDeadlines is a dictionary of task:deadline 
        #     # and this function Identify the task graphs which have tasks missing their 
        #     # deadlines, and select a task from these and move it to a PE with no deadline 
        #     # violation
        #     for task, deadline in taskDeadlines.items():
        #         if task in schedulesTasks:
        #             scheduled_time = schedulesTasks[task][0] + exec_time[tasks.index(task)][schedulesTasks[task][1]]
        #             if scheduled_time > deadline:
        #                 #logic to find a PE with no deadline violation
        #                 # and move the task there
        #                 ##################################
        #                 ##################################
        #                 ##################################
        #                 return task
            
            
class MyCrossover(Crossover):
    #Crossover on PE types and tasks mapped to PE. This operator
    #copies the mapping and PE-type from one individual to a PE in
    #another individual
    def __init__(self, num_pes=4, prob=0.4):
        super().__init__(2, 2)
        self.num_pes = int(num_pes)
        self.prob = prob
        
    def _do(self, problem, X, **kwargs):
        n_offsprings, n_matings, n_var = X.shape
        Y = np.full_like(X, np.nan)

        for i in range(n_matings):
            parent1 = X[0, i]
            parent2 = X[1, i]

            if np.random.rand() < self.prob:
                crossover_point = np.random.randint(1, n_var - 1)

                Y[0, i, :crossover_point] = parent1[:crossover_point]
                Y[0, i, crossover_point:] = parent2[crossover_point:]

                Y[1, i, :crossover_point] = parent2[:crossover_point]
                Y[1, i, crossover_point:] = parent1[crossover_point:]
            else:
                Y[0, i] = parent1
                Y[1, i] = parent2
            print("individual after crossover: ", Y)
        return Y
    





def printGraph(tasks, weights, exec_time, resource_graph):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()

    for task in tasks:
        G.add_node(task)

    for (src, dest), weight in weights.items():
        G.add_edge(src, dest, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Task Graph")
    plt.show()
    
    
    
printGraph(tasks = ["a", "b", "c", "d", "x"],
        
resource_graph = {"a" : "fpga", "b" : "gpp", "c" : "asic", "d" : "gpp", "x" : "fpga"},
        
weights = {
            ("a", "b"): 2,
            ("x", "b"): 3,
            ("a", "c"): 4,
            ("b", "c"): 5,
            ("b", "d"): 3,
            ("c", "d"): 1
        },
        
exec_time = [
            [0.9, 1.4, 0.7, 1.0],  # task a
            [1.1, 1.0, 0.6, 0.9],  # task b
            [0.8, 1.2, 0.9, 1.1],  # task c
            [1.3, 0.9, 0.7, 1.2],  # task d
            [0, 0, 0, 0],          # task x
        ])