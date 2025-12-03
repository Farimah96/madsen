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
        F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return

        self.ax.clear()
        self.ax.scatter(F[:,0], F[:,1], c='blue')
        self.ax.set_xlabel("Makespan")
        self.ax.set_ylabel("cost")
        self.ax.set_title(f"Pareto Front - Generation {algorithm.n_gen}")
        plt.pause(0.1)

 # Create  algorithm
algorithm = NSGA2(
    pop_size=20,
    sampling=MySampling(num_pes=len(problem.pe_types_inst)),
    crossover=MyCrossover(num_pes=len(problem.pe_types_inst)),
    mutation=MyMutation(num_pes=len(problem.pe_types_inst)),
    eliminate_duplicates=True
)

#optimization

termination = get_termination("n_gen", 15)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    verbose=True,
    callback=MyCallback(ax)
)

plt.ioff()
plt.show()

# Results

print("\nFinal Pareto Front (makespan, cost):\n", res.F)
print("\nBest chromosomes (PE assignment vectors):\n", res.X)
