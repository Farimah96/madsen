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


# Application Model
""" 1 task graph for simple mode """



class fixed_arch_problem(ElementwiseProblem):
    def __init__(self, elementwise=True, **kwargs):
        super().__init__(elementwise,n_var=4, n_obj=3, **kwargs)
        
        self.n_tasks = 4
        self.n_edges = 5
        
        tasks = ["task1", "task2", "task3", "task4"]
        
        e1 = 2
        e2 = 1
        e3 = 1
        e4 = 3
        e5 = 2
        
        
        self.edges = [
            (1,2,e1),
            (1,3,e2),
            (2,3,e3),
            (2,4,e4),
            (3,4,e5)
        ]
        
        self.pe_power = [2.2, 1.2, 1.8, 1.2]  # watts
        
        global exec_time
        exec_time = [
            [0.9, 1.4, 0.7, 1.4],  # task 0 on PEs 0..3
            [1.1, 1.0, 0.6, 1.0],
            [0.8, 1.2, 0.9, 1.2],
            [1.3, 0.9, 0.7, 0.9]
        ]
        
        global communication_time
        communication_time = 5
        
        global deadlines
        deadlines = [30, 20, 40, 60]
        periods = [30, 20, 20, 30]
        # execution time, ci, the consumed energy, ei, and the memory usage, mi, are determined by the actual mapping.
        
        
        pe_n = 4
        pe_types = ["fpga", "asic", "gpp"]
        bus_n = 2
        bridge_ = 1
        #ignore bus BW and bus power consumption
        
        # fixed architecture
        
        pes = [pe_types[0], pe_types[2], pe_types[1], pe_types[2]]
        
        buses = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        bridges = [1, 2]
        
        chromosome = [] # aka tasks here
        
    
    def static_list_scheduler(chromosome):
        t_level = []
        b_level = []
        
        # t_level computation
        for indeks in chromosome:
            x = indeks.index()
            if x == 0:
                t_level[0] = exec_time[0, indeks]
                return(t_level)
                
            else:
                for i in range(1, x):
                    t_level[indeks] = exec_time[indeks, indeks] + communication_time
                    i -= 1
                    if i == 1:
                        return t_level
            
        
        # b_level computation
        for bindeks in chromosome:
            x = bindeks.index()
            if x == -1:
                b_level[-1] = deadlines[-1] - exec_time[-1, bindeks]
                return(b_level)
                
            else:
                for i in range(x, -1):
                    b_level[bindeks] = exec_time[bindeks, bindeks] + communication_time
                    i += 1
                    if i == -1:
                        return(b_level)
        
    
        
        


        
    #def evaluate():   
        

# class my_sampling(Sampling):
#     def _do(self, problem, n_samples, **kwargs):
#         return np.random.randint(0, problem.n_pe, size=(n_samples, problem.n_tasks))