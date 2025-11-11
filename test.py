from madsen1 import fixed_arch_problem, MyMutation, MyCrossover, MySampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt

problem = fixed_arch_problem()

plt.ion()
fig, ax = plt.subplots()

class MyCallback(Callback):
    def __init__(self, ax):
        super().__init__()
        self.ax = ax

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        self.ax.clear()
        self.ax.scatter(F[:,0], F[:,1], c='blue')
        self.ax.set_xlabel("Makespan")
        self.ax.set_ylabel("Cost")
        self.ax.set_title(f"Pareto Front Generation {algorithm.n_gen}")
        plt.pause(0.1)

algorithm = NSGA2(
    pop_size=20,
    sampling=MySampling(),
    crossover=MyCrossover(),
    mutation=MyMutation(),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 10)
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True,
    callback=MyCallback(ax)
)

plt.ioff()
plt.show()

print("Final Pareto Front (makespan, cost):\n", res.F)
print("Best chromosomes:\n", res.X)
