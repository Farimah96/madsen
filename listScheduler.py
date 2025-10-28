class ListScheduler():
    
    def scheduler(tasks, tEdges, resouces, rEdges, allocation, priority):

        startList = []
        readyList = []
        t = 0
        scheduledTasks = {"a" : 0, "b" : 0, "c" : 0, "d" : 0, "x" : 0}
        exec_time = [
            [1, 2, 1, 2, 1],
            [2, 3, 2, 3, 2],
            [1, 2, 3, 1, 2]
        ]
        
        resouces = ["GPP", "FPGA", "ASIC"]
        allocation = {"GPP": 2, "FPGA": 1, "ASIC": 1}
        
        priorityList = []
        
        for task in tasks:
            if priority[task] == 0:
                readyList.append(task)
        
        while len(tasks) > 0:
            for resource in resouces:
            #for each resource r:
            #   remove finished tasks from r
            #   check which new tasks are now ready for r
            #   choose the best ones by priority (p)
            #   start them and set τ(v) = t    
                
                finishTimeOfTask = exec_time[resouces.index(resource)][tasks.index(task)] + scheduledTasks[task]
                if finishTimeOfTask == t:
                    tasks.remove(resource)
                    print(f"Task {resource} finished at time {t}")
                    
                    
                    
                    