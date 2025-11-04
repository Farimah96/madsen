from madsen1 import fixed_arch_problem
from madsen1 import MySampling
from madsen1 import Representation
from unittest.mock import Mock

# static_list_scheduler = fixed_arch_problem.static_list_scheduler
# static_list_scheduler(["a", "b", "c", "d", "x"], {("a" , "b"):2, ("x", "b"):3, ("a" , "c"):4, ("b" , "c"):5, ("b" , "d"):3, ("c" , "d"):1},
#                       resource_graph= {"a" : "fpga", "b" : "gpp", "c" : "asic", "d" : "gpp", "x" : "fpga"}, exec_time = [
#                         [0.9, 1.4, 0.7],  # task 0 on PEs 0..3
#                         [1.1, 1.0, 0.6],
#                         [0.8, 1.2, 0.9],
#                         [1.3, 0.9, 0.7],
#                         [0, 0, 0]
#                       ], communication_time=5, 
#                       pe_types=["fpga", "gpp", "asic", "gpp"]
#                       )


obj = Representation.encode([0, 2, 1, 0, 2], ["a", "b", "c", "d", "x"])
problem = Mock()
problem.n_var = 5
sampling = MySampling(num_pes=3, fixed_indices=None, fixed_values=None)
samples = sampling._do(problem=problem, n_samples=100)
print("Generated samples:\n", samples)



