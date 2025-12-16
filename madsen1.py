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

        # NOTE: representation keeps topology, but for fixed platform we will set these in problem
        # each row means a bus and each column means a PE
        # (kept here for reference; actual platform topology is defined in fixed_arch_problem)
        self.buses = [
            [1,1,0,0],
            [0,0,1,1]
        ]

        # each row means a bridge and each column means a bus (or representation of bridges)
        self.bridges = [
            [1, 2]   # example
        ]

    def encode(chromosome, tasks): #return dictionary of task:pe
        mapping = {}
        for i, task in enumerate(tasks):
            mapping[task] = chromosome[i]
        print("mapping is:", mapping)
        return mapping


class fixed_arch_problem(ElementwiseProblem):
    def __init__(self, elementwise=True, **kwargs):
        super().__init__(elementwise,n_var=5, n_obj=2, **kwargs)

        # tasks
        self.tasks = ["a", "b", "c", "d"]

        # NOTE: for FIXED platform the resource_graph for mapping SHOULD map task -> PE_INDEX
        # default initial mapping (example): here we keep a default type-map (not used for scheduling)
        # but platform description below defines PE types and topology.
        # We'll keep a convenience dict of PE types (index->type)
        self.pe_types = ["fpga", "gpp", "asic"]
        # actual instantiated PEs on the fixed platform (list of types by index)
        self.pe_types_inst = ["fpga", "gpp", "asic", "gpp"]  # example: 4 PEs indices 0..3

        # define platform topology (fixed). These belong to the platform description.
        # buses: each row is a bus, each column corresponds to a PE index:
        # buses[i][j] == 1 if PE j is connected to bus i
        self.buses = [
            [1,1,0,0],  # bus 0 connects PE0 and PE1
            [0,0,1,1],  # bus 1 connects PE2 and PE3
        ]
        # bridges: each row is a bridge and indicates which buses it connects (example)
        # here a bridge connecting bus0 and bus1
        self.bridges = [
            [1, 1]  # representation: connects bus0 and bus1 (example)
        ]

        # note: resource_graph (task -> pe index) will be given per individual (mapping)
        # but keep a default sample mapping (not used by GA)
        self.resource_graph = {"a" : 0, "b" : 1, "c" : 2, "d" : 1}

        self.weights = {
            ("a", "b"): 2,
            ("a", "c"): 4,
            ("b", "c"): 5,
            ("b", "d"): 3,
            ("c", "d"): 1
        }

        self.exec_time = [
            [0.9, 1.4, 0.7, 1.0],  # task a  (exec times for PE indices 0..3)
            [1.1, 1.0, 0.6, 0.9],  # task b
            [0.8, 1.2, 0.9, 1.1],  # task c
            [1.3, 0.9, 0.7, 1.2],  # task d
        ]

        # communication_time is not used directly as a constant in fixed-mode scheduler:
        # we'll keep it as a fallback but ECT should be determined from bus availabilities/bandwidths.
        self.communication_time = 5

    def static_list_scheduler(self, tasks, weights, resource_graph, exec_time, communication_time, pe_types, buses, bridges):

        # helper: parents & children in application graph (weights keys)
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

        # Topological list (Kahn)
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

        # compute t-level using mapping task->pe_index (resource_graph)
        def compute_t_level(TopList):
            # earliest start time of task considering parent completion + communication
            t_level = {task : 0 for task in TopList}

            for node in TopList:
                parents = return_parents(node)
                if not parents:
                    t_level[node] = 0
                    continue
                maxv = 0
                for parent in parents:
                    if parent in tasks:
                        # parent's finish = t_level[parent] + exec_time(parent on its assigned PE) + communication(parent->node)
                        parent_pe = resource_graph[parent]  # index
                        parent_exec = exec_time[tasks.index(parent)][parent_pe]
                        comm = weights.get((parent, node), 0)
                        candidate = t_level[parent] + parent_exec + comm
                        if candidate > maxv:
                            maxv = candidate
                t_level[node] = maxv

            return t_level

        # compute b-level using mapping task->pe_index (resource_graph)
        def compute_b_level(RevTopList):
            b_level = {task : 0 for task in RevTopList}
            for node in RevTopList:
                childrens = return_children(node)
                if not childrens:
                    b_level[node] = 0
                    continue
                maxv = 0
                for child in childrens:
                    if child in tasks:
                        child_pe = resource_graph[child]
                        child_exec = exec_time[tasks.index(child)][child_pe]
                        comm = weights.get((node, child), 0)
                        candidate = b_level[child] + child_exec + comm
                        if candidate > maxv:
                            maxv = candidate
                b_level[node] = maxv

            # keep same ordering as caller expects (reversed dict used earlier)
            b_level_rev = dict(reversed(list(b_level.items())))
            return b_level_rev

        def Computing_priorities(nodes, t_level, b_level):
            priority_list = {}
            for node in nodes:
                priority = b_level[node] + t_level[node]
                priority_list.update({node:priority})
            sorted_priority_list = dict(sorted(priority_list.items(), key=lambda item: item[1], reverse=True))
            return sorted_priority_list

        def NumـUnschedueldـPredecessors(tasks, weights):
            prec_dict = {task: 0 for task in tasks}
            prec_task_names = {task: [] for task in tasks}
            unscheduled_tasks = list(tasks)

            for edge in weights.keys():
                if edge[1] in prec_dict:  # only real tasks
                    prec_dict[edge[1]] += 1
                    prec_task_names[edge[1]].append(edge[0])

            return unscheduled_tasks

        # find earliest start time considering parents' finish times AND PE availability
        def find_est(task, pe, scheduled_tasks):
            # pe is PE index (int)
            if not scheduled_tasks:
                return 0
            else:
                maxv = 0
                parents = return_parents(task)
                # parent's finish times + communication
                for parent in parents:
                    if parent in scheduled_tasks.keys():
                        parent_pe = scheduled_tasks[parent][1]  # parent assigned PE index
                        parent_end_time = scheduled_tasks[parent][0] + exec_time[tasks.index(parent)][parent_pe]
                        # add communication if parent_pe != pe
                        comm_t = 0
                        if parent_pe != pe:
                            comm_t = weights.get((parent, task), 0)
                        candidate = parent_end_time + comm_t
                        if candidate > maxv:
                            maxv = candidate
                # also consider PE availability: if another task is running on 'pe', it may block
                for tname, info in scheduled_tasks.items():
                    assigned_pe = info[1]
                    if assigned_pe == pe:
                        t_end = info[0] + exec_time[tasks.index(tname)][assigned_pe]
                        if t_end > maxv:
                            maxv = t_end
                return maxv

        # condition_passed: parents scheduled AND PE is available at EST (we check conservatively)
        def condition_passed(task, scheduled_tasks, est):
            parents = [p for p in return_parents(task) if p in tasks]
            if not parents:
                # root ready
                return True

            # all parents must be scheduled
            for parent in parents:
                if parent not in scheduled_tasks:
                    return False

            # check PE availability: no running task on the same PE that overlaps est
            task_pe = resource_graph[task]
            for tname, info in scheduled_tasks.items():
                assigned_pe = info[1]
                if assigned_pe == task_pe:
                    t_end = info[0] + exec_time[tasks.index(tname)][assigned_pe]
                    if t_end > est:
                        return False

            return True


        print("scheduling with static list scheduling algorithm:\n")


        def final_scheduler(tasks, weights, resource_graph, pe_types, buses, bridges):
            scheduled_tasks = {} # task : (start_time, pe_index)
            unscheduled_tasks = NumـUnschedueldـPredecessors(tasks, weights)

            TopList = topologyList(tasks)
            RevTopList = reverse_topologyList(tasks)

            t_level = compute_t_level(TopList)
            b_level = compute_b_level(RevTopList)
            priority_list = Computing_priorities(tasks, t_level, b_level)

            timeline = []  # list of (task, pe_type, start, end)

            while unscheduled_tasks:
                # choose highest priority ready task
                y = None
                for task in priority_list.keys():
                    if task in unscheduled_tasks:
                        est_candidate = find_est(task, resource_graph[task], scheduled_tasks)
                        if condition_passed(task, scheduled_tasks, est_candidate):
                            y = task
                            break

                if y is None:
                    # no ready task found (should not happen in a DAG unless bug) -> break to avoid infinite loop
                    break

                pe_index = resource_graph.get(y)
                est = find_est(y, pe_index, scheduled_tasks)

                scheduled_tasks[y] = (est, pe_index) # (start_time, pe)
                unscheduled_tasks.remove(y)

            # print("final scheduled tasks are:", scheduled_tasks)

            # print scheduling details############################################################################################################################
            for task, (start, pe) in scheduled_tasks.items():
                exec_t = exec_time[tasks.index(task)][pe]
                pe_type = pe_types[pe] if pe < len(pe_types) else f"PE{pe}"
                # print(f"Task {task} is scheduled on PE {pe} (type {pe_type}) starting at time {start} with execution time {exec_t}")
                parents = return_parents(task)
                for parent in parents:
                    if parent in scheduled_tasks:
                        comm_t = weights.get((parent, task), 0)
                        if comm_t > 0:
                            continue
                            # print(f"  Communication time from parent task {parent} to task {task} is {comm_t}")

            # print all of chromosomes of each generation
            
            
            return scheduled_tasks


        # call final scheduler with topology
        return final_scheduler(tasks, weights, resource_graph, pe_types, buses, bridges)

    def _evaluate(self, x, out, *args, **kwargs):
        # IMPORTANT: for FIXED platform mapping must be task -> PE index (integer)
        mapping = { self.tasks[i] : int(x[i]) for i in range(len(self.tasks)) }

        sch_dict = self.static_list_scheduler(
            tasks=self.tasks,
            weights=self.weights,
            resource_graph=mapping,
            exec_time=self.exec_time,
            communication_time=self.communication_time,
            pe_types=self.pe_types_inst,   # use inst list for PE index -> type
            buses=self.buses,
            bridges=self.bridges
        )

        # cost: sum of cost per PE type used by tasks
        system_cost = {"fpga": 100, "gpp": 50, "asic": 150}
        cost = sum(system_cost[pe] for pe in self.pe_types_inst)

        print(cost)

        finish_time = 0
        for task_name, info in sch_dict.items():
            start = info[0]
            pe_index = info[1]
            task_exec = self.exec_time[self.tasks.index(task_name)][pe_index]
            if start + task_exec > finish_time:
                finish_time = start + task_exec

        print("Makespan is:", finish_time)
        print("cost is:", cost)
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

    def valid_sample(self, sample, resource_graph, pe_types): # sample is a list of pe indices
        for pe in sample:
            if pe < 0 or pe >= len(pe_types):
                return False
            # resource_graph here should be mapping task->pe_index; we cannot check types here reliably
            # keep minimal checks only
        return True

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

    # mutation: change assigned PE index for some tasks (fixed-platform style)
    def _do(self, problem, X, **kwargs):
        Y = np.full_like(X, np.nan)
        for i in range(X.shape[0]):
            individual = X[i].copy()
            for j in range(len(individual)):
                if np.random.rand() < self.mutation_rate:
                    new_pe = np.random.randint(0, self.num_pes)
                    individual[j] = new_pe
            Y[i] = individual
            # print("individual after mutation: ", Y[i])
        return Y


class MyCrossover(Crossover):
    # Crossover on mapping array (task -> PE index)
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
            # print("individual after crossover: ", Y[:, i, :])
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


# NOTE: This printGraph call is for displaying the application graph only (uses a literal resource_graph mapping to types)
printGraph(tasks = ["a", "b", "c", "d"],

resource_graph = {"a" : "fpga", "b" : "gpp", "c" : "asic", "d" : "gpp"},

weights = {
            ("a", "b"): 2,
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
        ])

