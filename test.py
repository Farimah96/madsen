from madsen1 import fixed_arch_problem, MyMutation, MyCrossover, MySampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt

# problem instance

problem = fixed_arch_problem()

plt.ion()
fig, ax = plt.subplots()

# Plotting

class MyCallback(Callback):
    def __init__(self, ax):
        super().__init__()
        self.ax = ax

    def notify(self, algorithm):
        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")

        if X is None or F is None:
            return

        print(f"\n===== Generation {algorithm.n_gen} =====")
        for i in range(len(X)):
            print(f"Chromosome {i}: {X[i].astype(int)} -> Objectives: {F[i]}")

        # ==================== Plot ====================
        self.ax.clear()

        self.ax.scatter(F[:, 0], F[:, 1], c='gray', s=50, label='Population')

        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        self.ax.scatter(F[nds_idx, 0], F[nds_idx, 1], c='red', s=80, label='Pareto Front')

        self.ax.set_xlabel("Makespan")
        self.ax.set_ylabel("Cost")
        self.ax.set_title(f"Pareto Front - Generation {algorithm.n_gen}")
        self.ax.legend()
        plt.pause(0.1)


 # Create  algorithm
algorithm = NSGA2(
    pop_size=10,
    sampling=MySampling(num_pes=len(problem.pe_types_inst)),
    crossover=MyCrossover(num_pes=len(problem.pe_types_inst)),
    mutation=MyMutation(num_pes=len(problem.pe_types_inst)),
    eliminate_duplicates=True,
    save_history=True
)

#optimization

termination = get_termination("n_gen", 30)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    verbose=True,
    callback=MyCallback(ax)
)

# # print all of chromosomes of each generation
# for gen, gen_data in enumerate(algorithm.history):
#     pop = gen_data.pop
#     print(f"\nGeneration {gen + 1} chromosomes and objectives:")
#     for ind in pop:
#         X = ind.X.astype(int)
#         F = problem.evaluate(X, return_values_of=["F"])["F"]
#         print(f"Chromosome: {X} -> Objectives: {F}")


plt.ioff()
plt.show()

# Results
print("\nBest chromosomes (PE assignment vectors):\n", res.X)
print("\nFinal Pareto Front (makespan, cost):\n", res.F)
