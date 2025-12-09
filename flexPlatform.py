"""
Flexible platform mapping implementation (Madsen-style chromosome).
Chromosome layout: [ allocation (n_types) | binding (n_tasks) ]
- allocation: integer counts per resource type
- binding: for each task, index of the target processing element instance (0..P-1)
This implementation provides:
- decode: allocation -> platform (list of PE types)
- repair: binding values mod P
- static list scheduler (list-scheduling using b-level priority)
- objectives: makespan and platform cost
- operators: sampling, crossover (two-stage), mutation (multiple modes: allocation change, add/remove PE, reassign, bus-aware reassign, heuristic placeholder)
- test bench using NSGA2 is provided below the classes (commented out for optional run)
Dependencies: pymoo, numpy, networkx (for optional graph plotting)
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

# Optional: plotting for debug
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    _HAS_NX = True
except Exception:
    _HAS_NX = False

################################################################################
#                              Problem definition                              #
################################################################################
class FlexibleArchProblem(ElementwiseProblem):
    """
    Flexible architecture mapping problem.
    """
    def __init__(self,
                 tasks=None,
                 weights=None,
                 pe_types=None,
                 pe_cost=None,
                 exec_time_table=None,
                 max_alloc_per_type=4,
                 elementwise=True,
                 **kwargs):

        # default application if not provided
        self.tasks = tasks if tasks is not None else ["a", "b", "c", "d"]
        self.n_tasks = len(self.tasks)

        # communication weights between tasks (edge weights)
        self.weights = weights if weights is not None else {
            ("a", "b"): 2,
            ("a", "c"): 4,
            ("b", "c"): 5,
            ("b", "d"): 3,
            ("c", "d"): 1
        }

        # resource types and costs
        self.pe_types = pe_types if pe_types is not None else ["fpga", "gpp", "asic"]
        self.n_types = len(self.pe_types)
        self.pe_cost = pe_cost if pe_cost is not None else {"fpga":100, "gpp":50, "asic":150}

        # execution time table indexed by task index and type index
        # exec_time_table[task_index][type_index]
        if exec_time_table is not None:
            self.exec_time_table = exec_time_table
        else:
            # default: times for types in same order as pe_types
            self.exec_time_table = [
                [0.9, 1.4, 0.7],  # a
                [1.1, 1.0, 0.6],  # b
                [0.8, 1.2, 0.9],  # c
                [1.3, 0.9, 0.7],  # d
            ]

        # upper bound for allocation counts per type for sampling
        self.max_alloc = int(max_alloc_per_type)

        # chromosome length: allocation part (n_types) + binding part (n_tasks)
        n_var = self.n_types + self.n_tasks
        # two objectives: makespan (min) and cost (min)
        super().__init__(elementwise=elementwise, n_var=n_var, n_obj=2, **kwargs)

    # -------------------------
    #      decode helpers     #
    # -------------------------
    def allocation_to_platform(self, alloc_vector):
        """
        Build platform instance list from allocation vector.
        Returns list of PE types in instance order (indexable by instance index).
        """
        platform = []
        for t_index, cnt in enumerate(alloc_vector):
            if cnt is None:
                continue
            for _ in range(int(max(0, cnt))):
                platform.append(self.pe_types[t_index])
        return platform
        # in => alloc_vector = [A, B, C, ...] & out => platform = ["cpu", "cpu", "cpu", "dsp", "asic", ...]  // based on pe types index

    def build_buses_simple(self, platform):
        """
        Build a simple bus partitioning for the platform.
        This is lightweight: it partitions platform instances into two buses (first half, second half).
        Returns: buses (list of lists of PE indices), and bridges (list of tuples connecting bus indices).
        This is sufficient to implement bus-aware reassign mutation.
        """
        P = len(platform)
        if P == 0:
            return [[]], []
        half = max(1, P // 2)
        buses = [list(range(0, half)), list(range(half, P))] if P > 1 else [list(range(P))]
        # clean empty
        buses = [b for b in buses if b]
        bridges = []
        if len(buses) > 1:
            # connect bus0 and bus1
            bridges.append((0,1))
        return buses, bridges
        # ex:platform = ["cpu", "cpu", "cpu", "dsp", "asic", "asic"]
        # half = P // 2 = 3
        # bus0 = [0, 1, 2]
        # bus1 = [3, 4, 5]
        # buses = [
        #     [0, 1, 2],   # bus0
        #     [3, 4, 5]    # bus1
        # ]
        # bridges = [(0, 1)]



    def binding_to_mapping(self, platform, binding):
        """
        Convert binding genes (raw integers) to a mapping task -> PE_index.
        Repair binding values modulo P (number of instances).
        """
        mapping = {}
        P = len(platform)
        if P == 0:
            # degenerate: no platform instances, assign virtual 0
            for i, task in enumerate(self.tasks):
                mapping[task] = 0
            return mapping

        for i, task in enumerate(self.tasks):
            raw = int(binding[i])
            # repair: modulo P
            mapping[task] = raw % P
        return mapping
    
        # platform = ["cpu", "cpu", "cpu", "dsp", "asic", "asic"]
        # binding  = [5, 8, 11, 0]   # raw genes
        # tasks    = ["a", "b", "c", "d"]
        # P = 6 
        # mapping[task] = binding[i] % P
        # { "a": 5, "b": 2, "c": 5, "d": 0 }

    # --------------------------------------------------
    #    scheduler (static list scheduling variant)1    #
    # --------------------------------------------------
    # def static_list_scheduler(self, tasks, weights, mapping, platform):
        """
        List-scheduling using b-level priority (as in many HEFT-like static list methods).
        mapping: dict task->PE_index
        exec times: derived from exec_time_table using platform[pe_index] -> type index
        """
        # helper parent/children lists
        parents = {t: [] for t in tasks}
        children = {t: [] for t in tasks}
        for (u,v), w in weights.items():
            if u in tasks and v in tasks:
                parents[v].append(u)
                children[u].append(v)

        # topo order Kahn
        indeg = {t:0 for t in tasks}
        for v in tasks:
            indeg[v] = len(parents[v])
        Q = [t for t in tasks if indeg[t] == 0]
        topo = []
        while Q:
            x = Q.pop(0)
            topo.append(x)
            for c in children[x]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    Q.append(c)

        # determine exec_time(task, pe_index)
        def exec_time_on_pe(task, pe_index):
            # find type index of platform[pe_index]
            if pe_index < 0 or pe_index >= len(platform):
                return 1e6
            ptype = platform[pe_index]
            type_index = self.pe_types.index(ptype)
            return self.exec_time_table[self.tasks.index(task)][type_index]

        # compute b-level
        b_level = {t: 0 for t in tasks}
        for t in reversed(topo):
            best = 0
            for c in children[t]:
                comm = self.weights.get((t,c), 0)
                # child exec on its assigned PE
                child_pe = mapping[c]
                child_exec = exec_time_on_pe(c, child_pe)
                val = b_level[c] + comm + child_exec
                if val > best:
                    best = val
            b_level[t] = best

        priority_list = sorted(tasks, key=lambda x: b_level[x], reverse=True)

        scheduled = {}  # task -> (start_time, pe_index)
        unscheduled = set(tasks)

        # helper to compute earliest start time
        def find_est(task, pe, scheduled):
            # consider parent finish + communication if on different PE
            est = 0
            for p in parents[task]:
                if p in scheduled:
                    p_start, p_pe = scheduled[p]
                    p_end = p_start + exec_time_on_pe(p, p_pe)
                    comm = self.weights.get((p, task), 0) if p_pe != pe else 0
                    est = max(est, p_end + comm)
                else:
                    # parent not scheduled yet -> not ready
                    est = max(est, 0)
            # consider PE availability
            for tname, (s, ppe) in scheduled.items():
                if ppe == pe:
                    t_end = s + exec_time_on_pe(tname, ppe)
                    if t_end > est:
                        est = t_end
            return est

        # scheduling loop
        while unscheduled:
            chosen = None
            for t in priority_list:
                if t not in unscheduled: 
                    continue
                # check ready: all parents scheduled
                ready = all(p in scheduled for p in parents[t])
                if not ready:
                    continue
                pe = mapping[t]
                est_candidate = find_est(t, pe, scheduled)
                chosen = t
                break
            if chosen is None:
                # no ready task found (cycle or issue). break to avoid infinite loop
                break
            est = find_est(chosen, mapping[chosen], scheduled)
            scheduled[chosen] = (est, mapping[chosen])
            unscheduled.remove(chosen)

        return scheduled

    # --------------------------------------------------
    #    scheduler (static list scheduling variant)2    #
    # --------------------------------------------------    

    def static_list_scheduler(self, tasks, weights, mapping, platform):
        """
        HEFT-based static list scheduling for flexible platform.
        Inputs:
            tasks: ordered list of task names
            weights: dict { (parent,child): communication_cost }
            mapping: dict { task -> PE_index_from_chromosome }
            platform: list of PE types (e.g., ["fpga","asic","gpp",...])
        Output:
            scheduled_tasks = { task: (start_time, pe_index) }
        """

        # -------------------------
        # Build parents / children
        # -------------------------
        parents = {t: [] for t in tasks}
        children = {t: [] for t in tasks}
        for (u, v), w in weights.items():
            if u in tasks and v in tasks:
                parents[v].append(u)
                children[u].append(v)

        # -------------------------
        # Topological ordering
        # -------------------------
        indeg = {t: len(parents[t]) for t in tasks}
        Q = [t for t in tasks if indeg[t] == 0]
        topo = []
        while Q:
            x = Q.pop(0)
            topo.append(x)
            for c in children[x]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    Q.append(c)

        # reverse topo
        rev_topo = list(reversed(topo))

        # -------------------------
        # Exec time lookup
        # -------------------------
        def exec_time_on_pe(task, pe_index):
            """Return execution time of a task on a PE index. If invalid mapping → return huge penalty."""
            if pe_index < 0 or pe_index >= len(platform):
                return 1e6  # penalize invalid PE
            ptype = platform[pe_index]
            type_index = self.pe_types.index(ptype)
            return self.exec_time_table[self.tasks.index(task)][type_index]

        # -------------------------
        # Compute t-level (as you wanted, identical formula)
        # -------------------------
        t_level = {t: 0 for t in tasks}
        for node in topo:
            maxv = 0
            for p in parents[node]:
                p_pe = mapping[p]
                p_exec = exec_time_on_pe(p, p_pe)
                comm = weights.get((p, node), 0)
                finish_p = t_level[p] + p_exec + comm
                maxv = max(maxv, finish_p)
            t_level[node] = maxv

        # -------------------------
        # Compute b-level (reverse topo)
        # -------------------------
        b_level = {t: 0 for t in tasks}
        for node in rev_topo:
            best = 0
            for c in children[node]:
                c_pe = mapping[c]
                c_exec = exec_time_on_pe(c, c_pe)
                comm = weights.get((node, c), 0)
                val = b_level[c] + c_exec + comm
                best = max(best, val)
            b_level[node] = best

        # -------------------------
        # Priority = t_level + b_level
        # -------------------------
        priority_list = sorted(tasks, key=lambda x: t_level[x] + b_level[x], reverse=True)

        # -------------------------
        # Helper: earliest start time
        # -------------------------
        def find_est(task, pe, scheduled):
            est = 0
            # parent finish constraints
            for p in parents[task]:
                if p in scheduled:
                    p_start, p_pe = scheduled[p]
                    p_end = p_start + exec_time_on_pe(p, p_pe)
                    comm = weights.get((p, task), 0) if p_pe != pe else 0
                    est = max(est, p_end + comm)
            # PE availability constraint
            for tname, (s, assigned_pe) in scheduled.items():
                if assigned_pe == pe:
                    t_end = s + exec_time_on_pe(tname, assigned_pe)
                    est = max(est, t_end)
            return est

        # -------------------------
        # Scheduling loop (HEFT)
        # -------------------------
        scheduled = {}
        unscheduled = set(tasks)

        while unscheduled:
            chosen = None
            for t in priority_list:
                if t not in unscheduled:
                    continue
                # check if all parents scheduled
                if not all(p in scheduled for p in parents[t]):
                    continue
                chosen = t
                break

            if chosen is None:
                break  # degeneracy protection

            pe = mapping[chosen]
            est = find_est(chosen, pe, scheduled)
            scheduled[chosen] = (est, pe)
            unscheduled.remove(chosen)

        return scheduled
        # return value is task_name : (start_time, pe_index)

    
    # -------------------------
    #        evaluation       #
    # -------------------------
    def _evaluate(self, x, out, *args, **kwargs):
        # x is array of length n_types + n_tasks
        alloc = np.array(x[:self.n_types], dtype=int)
        binding = np.array(x[self.n_types:], dtype=int)

        # build platform (list of PE types)
        platform = self.allocation_to_platform(alloc)

        # build buses (simple partition) for later use (not used by scheduler here)
        buses, bridges = self.build_buses_simple(platform)

        # build mapping (task -> pe index) with repair
        mapping = self.binding_to_mapping(platform, binding)

        # run scheduler
        schedule = self.static_list_scheduler(self.tasks, self.weights, mapping, platform)

        # compute makespan
        makespan = 0.0
        for t, (s, pe) in schedule.items():
            # task execution time according to platform[pe]
            if pe < 0 or pe >= len(platform):
                et = 1e6
            else:
                et = self.exec_time_table[self.tasks.index(t)][self.pe_types.index(platform[pe])]
            makespan = max(makespan, s + et)

        # compute cost: sum cost of platform instances
        cost = 0
        for p in platform:
            cost += self.pe_cost[p]

        out["F"] = np.array([makespan, cost])

################################################################################
#                   Sampling for flexible chromosome                           #
################################################################################
class FlexibleSampling(Sampling):
    """
    Sampling for flexible chromosomes.
    - allocation: random integers in [1..max_alloc]
    - binding: random integers in [0..P-1] where P = sum(allocation)
    """
    def __init__(self, n_types, n_tasks, max_alloc=4):
        super().__init__()
        self.n_types = n_types
        self.n_tasks = n_tasks
        self.max_alloc = int(max_alloc)

    def _do(self, problem, n_samples, **kwargs):
        pop = []
        for _ in range(n_samples):
            alloc = np.random.randint(1, self.max_alloc + 1, size=self.n_types)
            P = int(np.sum(alloc))
            if P <= 0:
                P = 1
            bind = np.random.randint(0, P, size=self.n_tasks)
            chrom = np.concatenate([alloc, bind]).astype(int)
            pop.append(chrom)
        return np.array(pop)

################################################################################
# Crossover: two-stage (allocation crossover then binding inheritance + repair)#
################################################################################
class FlexibleCrossover(Crossover):
    """
    Two-stage crossover:
    1) crossover allocation vectors (one-point)
    2) for each task, randomly inherit binding from one of the parents, then repair binding modulo P
    """
    def __init__(self, n_types, n_tasks, prob=0.8):
        super().__init__(2, 2)
        self.n_types = n_types
        self.n_tasks = n_tasks
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        n_off, n_mat, n_var = X.shape
        Y = np.full_like(X, np.nan)

        for k in range(n_mat):
            p1 = X[0, k].copy().astype(int)
            p2 = X[1, k].copy().astype(int)
            c1 = p1.copy()
            c2 = p2.copy()

            if np.random.rand() < self.prob:
                # allocation one-point crossover
                if self.n_types > 1:
                    point = np.random.randint(1, self.n_types)
                else:
                    point = 1
                c1[:point] = p1[:point]
                c1[point:self.n_types] = p2[point:self.n_types]
                c2[:point] = p2[:point]
                c2[point:self.n_types] = p1[point:self.n_types]

                # binding inheritance per-task
                for j in range(self.n_tasks):
                    if np.random.rand() < 0.5:
                        c1[self.n_types + j] = p1[self.n_types + j]
                        c2[self.n_types + j] = p2[self.n_types + j]
                    else:
                        c1[self.n_types + j] = p2[self.n_types + j]
                        c2[self.n_types + j] = p1[self.n_types + j]

                # repair binding according to new allocations
                for child in (c1, c2):
                    alloc_child = child[:self.n_types].astype(int)
                    P = int(np.sum(alloc_child))
                    if P <= 0:
                        P = 1
                    for j in range(self.n_tasks):
                        child[self.n_types + j] = int(child[self.n_types + j]) % P
            else:
                c1 = p1
                c2 = p2

            Y[0, k] = c1
            Y[1, k] = c2

        return Y

################################################################################
#                  Mutation: multi-mode mutation per article                   #
################################################################################
class FlexibleMutation(Mutation):
    """
    Mutation supports:
    - allocation increment/decrement (change counts)
    - addPE / removePE implemented as increment/decrement of allocation counts
    - random reassign tasks (binding mutation)
    - bus-aware reassign (tasks moved between PEs on same bus)
    - heuristic placeholder (if deadlines provided)
    """
    def __init__(self, n_types, n_tasks, mutation_rate=0.6, max_alloc=4):
        super().__init__()
        self.n_types = n_types
        self.n_tasks = n_tasks
        self.mrate = float(mutation_rate)
        self.max_alloc = int(max_alloc)

    def _do(self, problem, X, **kwargs):
        Y = X.copy().astype(int)
        for i in range(Y.shape[0]):
            ind = Y[i]

            # pick which mutation group to run (allow multiple small changes)
            # allocation changes
            for t in range(self.n_types):
                if np.random.rand() < (self.mrate / 2):
                    # increase or decrease allocation count
                    delta = np.random.choice([-1, 1])
                    ind[t] = int(max(1, min(self.max_alloc, ind[t] + delta)))

            # optional addPE (increment some allocation)
            if np.random.rand() < (self.mrate * 0.4):
                idx = np.random.randint(0, self.n_types)
                ind[idx] = int(min(self.max_alloc, ind[idx] + 1))

            # optional removePE (decrement some allocation if >1)
            if np.random.rand() < (self.mrate * 0.4):
                idx = np.random.randint(0, self.n_types)
                ind[idx] = int(max(1, ind[idx] - 1))

            # rebuild platform size and simple buses for binding mutation
            alloc = ind[:self.n_types]
            P = int(np.sum(alloc))
            if P <= 0:
                P = 1

            # binding mutation: random reassign some tasks
            for j in range(self.n_tasks):
                if np.random.rand() < self.mrate:
                    ind[self.n_types + j] = np.random.randint(0, P)

            # bus-aware reassign: move some tasks within same bus
            if np.random.rand() < (self.mrate * 0.5):
                # build simple buses: first half, second half
                buses = []
                if P == 1:
                    buses = [[0]]
                else:
                    half = max(1, P // 2)
                    buses = [list(range(0, half)), list(range(half, P))]
                # pick a bus with at least 1 PE
                bidx = np.random.randint(0, len(buses))
                src_bus = buses[bidx]
                if len(src_bus) >= 2:
                    # choose a PE from this bus and a different PE in same bus
                    pe_from = np.random.choice(src_bus)
                    pe_to = np.random.choice([p for p in src_bus if p != pe_from])
                    # pick some tasks currently assigned to pe_from and move them
                    assigned_tasks = [j for j in range(self.n_tasks) if (ind[self.n_types + j] % P) == pe_from]
                    if assigned_tasks:
                        k = min(len(assigned_tasks), np.random.randint(1, 4))
                        to_move = np.random.choice(assigned_tasks, size=k, replace=False)
                        for tm in to_move:
                            ind[self.n_types + tm] = pe_to

            # heuristic reassignment placeholder (no deadlines implemented)
            # if problem has attribute task_deadlines (dict), we could implement
            # moving tasks that miss deadlines to less loaded PEs.
            if hasattr(problem, "task_deadlines") and np.random.rand() < (self.mrate * 0.2):
                # basic heuristic: move one random task to least loaded PE
                loads = [0] * P
                for j in range(self.n_tasks):
                    loads[ind[self.n_types + j] % P] += 1
                least = int(np.argmin(loads))
                tmove = np.random.randint(0, self.n_tasks)
                ind[self.n_types + tmove] = least

        return Y

################################################################################
# Optional: test bench showing how to run NSGA2 (commented for import safety)  #
################################################################################
if __name__ == "__main__":
    # Quick test run with NSGA2 (requires pymoo installed)
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.core.callback import Callback
    import matplotlib.pyplot as plt

    problem = FlexibleArchProblem()

    sampling = FlexibleSampling(n_types=problem.n_types, n_tasks=problem.n_tasks, max_alloc=3)
    crossover = FlexibleCrossover(n_types=problem.n_types, n_tasks=problem.n_tasks, prob=0.9)
    mutation = FlexibleMutation(n_types=problem.n_types, n_tasks=problem.n_tasks, mutation_rate=0.3, max_alloc=3)

    algorithm = NSGA2(pop_size=100,
                      sampling=sampling,
                      crossover=crossover,
                      mutation=mutation,
                      eliminate_duplicates=True)

    # plotting callback
    plt.ion()
    fig, ax = plt.subplots()
    class PlotCallback(Callback):
        def __init__(self, ax):
            super().__init__()
            self.ax = ax
        def notify(self, algorithm):
            F = algorithm.pop.get("F")
            if F is None or len(F)==0:
                return
            self.ax.clear()
            self.ax.scatter(F[:,0], F[:,1])
            self.ax.set_xlabel("Makespan")
            self.ax.set_ylabel("Cost")
            self.ax.set_title(f"Generation {algorithm.n_gen}")
            plt.pause(0.01)

    termination = get_termination("n_gen", 30)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   callback=PlotCallback(ax),
                   verbose=True)

    plt.ioff()
    plt.show()

    print("Final Pareto front (makespan, cost):")
    print(res.F)
    print("Sample solution (allocation | binding):")
    print(res.X[0])
