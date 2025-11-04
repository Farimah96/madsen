from listScheduler import ListScheduler

# inputs

tasks = ["a", "b", "c", "d", "x"]
tEdges = {("a","b") : 0, ("a","c") : 0, ("b","c") : 0, ("c","d") : 0, ("b","d") : 0, ("x","b") : 0}

resouces = {"GPP", "FPGA", "ASIC"}
rEdges = {
    "GPP": ["b", "c", "d"],
    "FPGA": ["a", "c"],
    "ASIC": ["x", "c", "d"]
}

allocation = {"GPP": 2, "FPGA": 1, "ASIC": 1}

priority = {"a" : 0 , "x" : 0 , "b" : 1 , "c" : 2 , "d" : 3 }

ListScheduler.scheduler(tasks, tEdges, resouces, rEdges, allocation, priority)



