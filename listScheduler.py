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
                
        # for task in tasks:
        #     if priority[task] == 0:
        #         readyList.append(task)
        
        while len(tasks) > 0:
            for resource in resouces:
            #for each resource r:
            #   remove finished tasks from r
            #   check which new tasks are now ready for r
            #   choose the best ones by priority (p)
            #   start them and set τ(v) = t
            
            # use map() for mapping each task in list of value of resource edges
                mappedTasks = list(map(lambda x: x in rEdges[resource], tasks))
                availableSlots = allocation[resource]
                for i in range(len(mappedTasks)):
                    if mappedTasks[i] and availableSlots > 0 and tasks[i] in readyList:
                        taskToSchedule = tasks[i]
                        print(f"At time {t}, scheduling task {taskToSchedule} on resource {resource}")
                        scheduledTasks[taskToSchedule] = t
                        availableSlots -= 1
                        readyList.remove(taskToSchedule)
            t += 1
            # after scheduling, check for newly ready tasks
            for task in tasks:
                predecessors = [edge[0] for edge in tEdges.keys() if edge[1] == task]
                if all(scheduledTasks[pred] > 0 for pred in predecessors) and task not in readyList and scheduledTasks[task] == 0:
                    readyList.append(task)
                    
                    
        print(scheduledTasks)
                        