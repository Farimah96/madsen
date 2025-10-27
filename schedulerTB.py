from madsen1 import fixed_arch_problem

static_list_scheduler = fixed_arch_problem.static_list_scheduler
static_list_scheduler(["a", "b", "c", "d", "x"], {("a" , "b"):2, ("x", "b"):3, ("a" , "c"):4, ("b" , "c"):5, ("b" , "d"):3, ("c" , "d"):1},
                      resource_graph= {"a" : "fpga", "b" : "gpp", "c" : "asic", "d" : "gpp", "x" : "fpga"}, exec_time = [
                        [0.9, 1.4, 0.7],  # task 0 on PEs 0..3
                        [1.1, 1.0, 0.6],
                        [0.8, 1.2, 0.9],
                        [1.3, 0.9, 0.7],
                        [0, 0, 0]
                      ], communication_time=5, 
                      pe_types={"fpga": 1, "asic": 1, "gpp": 1}
                      )