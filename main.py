import math
import random
import copy
import time
# import matplotlib.pyplot as plt

class Task:
    def __init__(self, id, size, output, k,affinitive_machines):
        self.id = id
        self.size = size
        self.output = output
        self.num_machines = k
        self.affinitive_machines = [x - 1 for x in affinitive_machines]
        self.env_dep=set()  #该task依赖于env_dep中的task
        self.data_dep=set()
        self.machine_id=-1
        self.disk_id=-1
        self.a=0
        self.b=math.inf
        self.c=math.inf
        self.d=math.inf
        
class Machine:
    def __init__(self, id, power):
        self.id = id
        self.power = power
        self.time=0

class Disk:
    def __init__(self, id, speed, quota):
        self.id = id
        self.speed = speed
        self.quota = quota
        self.empty=quota
        # self.time=0

def load_data():
    
    
    l=int(input())
    tasks=[]
    for i in range(l):
        task_id, task_size, output_size, k, *affinitive_machines=list(map(int, input().split()))    
        tasks.append(Task(task_id, task_size, output_size, k, affinitive_machines))
        
    m=int(input())
    machines=[]
    for i in range(m):
        machine_id, power=list(map(int, input().split()))
        machines.append(Machine(machine_id, power))
        
    d=int(input())   
    disks=[]
    for i in range(d):
        disk_id, speed, quota=list(map(int, input().split()))
        disks.append(Disk(disk_id, speed, quota))
    
    a=int(input())
    for i in range(a):
        x,y=list(map(int, input().split()))
        tasks[y-1].data_dep.add(x-1)
        
    b=int(input())
    for i in range(b):
        x,y=list(map(int, input().split()))
        tasks[y-1].env_dep.add(x-1)
        
        
    return tasks, machines, disks

class TaskGraph:
    def __init__(self, tasks):
        self.num_tasks = len(tasks)
        self.tasks = tasks
        self.adj_list = [[] for _ in range(self.num_tasks)]
        self.in_degree = [0] * self.num_tasks
        for idx, task in enumerate(self.tasks):
            for dep_task_id in task.env_dep:
                self.adj_list[dep_task_id].append(idx)
                self.in_degree[idx] += 1
            for dep_task_id in task.data_dep:
                if dep_task_id not in task.env_dep:  # 避免重复计算入度
                    self.adj_list[dep_task_id].append(idx)
                    self.in_degree[idx] += 1
    
    def topological_sort(self):
        result_order = []
        zero_in_degree_tasks = [idx for idx in range(self.num_tasks) if self.in_degree[idx] == 0]
        while zero_in_degree_tasks:
            task_idx_to_remove = random.choice(zero_in_degree_tasks)
            result_order.append(task_idx_to_remove)
            zero_in_degree_tasks.remove(task_idx_to_remove)
            for neighbor_idx in self.adj_list[task_idx_to_remove]:
                self.in_degree[neighbor_idx] -= 1
                if self.in_degree[neighbor_idx] == 0:
                    zero_in_degree_tasks.append(neighbor_idx)
            
            #print(111)
       
        if len(result_order) != self.num_tasks:
            print("zero_in_degree_tasks=",zero_in_degree_tasks)
            print("result_order=",result_order)
            print("topological_sort error")
            return None  
        return result_order

class SA:
    class Individual:
        def __init__(self,num_tasks):
            self.tasks_plan=[-1]*num_tasks
            self.machines_plan=[-1]*num_tasks
            self.disks_plan=[-1]*num_tasks
            self.cost=0
            self.error=0
            self.mode=-1

    def __init__(self, tasks, machines, disks,  pop_size=50, init_temperature=1000,cooling_rate=0.95):
        self.tasks = tasks
        self.machines = machines
        self.disks = disks
        self.pop_size = pop_size
        self.init_temperature=init_temperature
        self.cooling_rate=cooling_rate
        self.num_tasks = len(tasks)
        self.num_machines = len(machines)
        self.num_disks = len(disks)
        self.record_cost = []
        self.record_error = []
        self.population = self.initialize_population()
        self.best_solution=copy.deepcopy(self.population[0])
        self.best_solution.cost = float('inf')
        
        
        # for individual in self.population:
        #     self.calculate_cost(individual)
            #print("cost=",individual.cost,"error=",individual.error)
    
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            sort_task=TaskGraph(self.tasks)
            disk_empty = [disk.quota for disk in self.disks]
            individual = SA.Individual(self.num_tasks)
            individual.tasks_plan = sort_task.topological_sort()
            for i in range(self.num_tasks):
                task=self.tasks[individual.tasks_plan[i]]
                individual.machines_plan[i] = random.choice(task.affinitive_machines)
                individual.disks_plan[i] = random.randint(0,self.num_disks-1)
                # chosen_disk=random.randint(0,self.num_disks-1)
                # while task.output > disk_empty[chosen_disk]:
                #      chosen_disk = random.randint(0, self.num_disks - 1)
                #      #print("task.output=",task.output,"chosen_disk=",chosen_disk,"disk_empty[chosen_disk]",disk_empty[chosen_disk])
                # individual.disks_plan[i] = chosen_disk
                # disk_empty[chosen_disk] -= task.output
            population.append(individual)
            #if _==1:
                # print("tasks_plan=",individual.tasks_plan)
                # print("machines_plan",individual.machines_plan)
                # print("disk_plan=",individual.disks_plan)
        return population
    
    def neighbor_solution(self,current_solution):
        new_solution=copy.deepcopy(current_solution)
        num_tasks = self.num_tasks
        idx1, idx2 = random.sample(range(num_tasks), 2)
        mode=random.randint(1,3)
        if(mode==1):
            new_solution.tasks_plan[idx1], new_solution.tasks_plan[idx2] = new_solution.tasks_plan[idx2], new_solution.tasks_plan[idx1]
            new_solution.machines_plan[idx1], new_solution.machines_plan[idx2] = new_solution.machines_plan[idx2], new_solution.machines_plan[idx1]
            new_solution.disks_plan[idx1], new_solution.disks_plan[idx2] = new_solution.disks_plan[idx2], new_solution.disks_plan[idx1]
        elif(mode==2):
            task=self.tasks[new_solution.tasks_plan[idx1]]
            new_solution.machines_plan[idx1]=random.choice(task.affinitive_machines)
        elif(mode==3):
            if(random.random()<=0.5):
                new_solution.disks_plan[idx1]=random.randint(0,self.num_disks-1)
            else:
                new_solution.disks_plan[idx1], new_solution.disks_plan[idx2] = new_solution.disks_plan[idx2], new_solution.disks_plan[idx1]
        return new_solution

    
    def calculate_cost(self,solution):
        global count1
        global count2
        finish_time=0
        count=0
        temp_tasks=copy.deepcopy(self.tasks)
        temp_machines=copy.deepcopy(self.machines)
        temp_disks=copy.deepcopy(self.disks)
        for i in range(self.num_tasks):
            # print("~"*50)
            task=temp_tasks[solution.tasks_plan[i]]
            # print("task_id=",task.id)
            task.machine_id=solution.machines_plan[i]
            # print(task.machine_id)
            machine = temp_machines[solution.machines_plan[i]]
            # print("machine.id=",machine.id," time=",machine.time)
            task.disk_id=solution.disks_plan[i]
            
            disk=temp_disks[solution.disks_plan[i]]
            # print("disk_id=",disk.id, "quota=",disk.quota)
            now_time=machine.time
            for j in task.data_dep:
                now_time=max(now_time,temp_tasks[j].d)
            # print("now_time=",now_time)
            task.a=now_time
            spend_time=0
            # print("task.data_dep=",task.data_dep)
            for j in task.data_dep:
                
                spend_time+=round(temp_tasks[j].output/temp_disks[temp_tasks[j].disk_id].speed,0)
            task.b=now_time+spend_time
            spend_time+=round(task.size/machine.power,0)
            task.c=now_time+spend_time
            spend_time+=round(task.output/disk.speed,0)
            task.d=now_time+spend_time
            machine.time=task.d
            
            disk.quota-=task.output
            finish_time=max(finish_time,task.d)
        error=self.check_error(temp_tasks,temp_machines,temp_disks)
        solution.cost=finish_time+error
        solution.error=error
        if error==0:
            count+=1
        del temp_tasks,temp_machines,temp_disks

    def check_error(self,tasks, machines, disks):
        quota_error=0
        env_error=0
        data_error=0
        for i in tasks:
            for j in i.env_dep:
                if i.a<tasks[j].c:
                    env_error+=abs(i.a-tasks[j].c)
            for j in i.data_dep:
                if i.a<tasks[j].d:
                    data_error+=abs(i.a-tasks[j].d)
        for i in disks:
            if i.quota<0:
                quota_error+=abs(i.quota)
        return env_error*150+round(data_error*160,0)+quota_error*150
    
    def acceptance_probability(self, cost, new_cost, temperature):
        if new_cost < cost:
            return 1.0
        else:
            return math.exp((cost - new_cost) / temperature)

    def solver(self,max_iteration=1400):
        current_temperature=self.init_temperature
        for iteration in range(max_iteration):
            new_population=[]
            for solution in self.population:
                
                new_solution=self.neighbor_solution(solution)
                self.calculate_cost(new_solution)
                self.calculate_cost(solution)
                if new_solution.error==0 and new_solution.cost<self.best_solution.cost:
                    self.best_solution=new_solution
                    
                if random.random() < self.acceptance_probability(solution.cost, new_solution.cost, current_temperature):
                    new_population.append(new_solution)
                   
                        
                else:          
                    new_population.append(solution)
                    
            current_temperature=self.cooling_rate*current_temperature
            self.population=new_population
            self.population.sort(key=lambda x: x.cost)
            self.record_cost.append(self.population[0].cost)
            self.record_error.append(self.population[0].error)
    def display(self):
       
        temp_tasks=copy.deepcopy(self.tasks)
        temp_machines=copy.deepcopy(self.machines)
        temp_disks=copy.deepcopy(self.disks)
        # for solution in self.population:
        #     if solution.error==0:
        #         solution=self.population[0]
        solution=copy.deepcopy(self.best_solution)
        finish_time=0
        for i in range(self.num_tasks):
                    task=temp_tasks[solution.tasks_plan[i]]
                    task.machine_id=solution.machines_plan[i]
                    machine = temp_machines[solution.machines_plan[i]]
                    task.disk_id=solution.disks_plan[i]
                    disk=temp_disks[solution.disks_plan[i]]
                    now_time=machine.time
                    task.a=now_time
                    spend_time=0
                    for j in task.data_dep:
                        
                        spend_time+=round(temp_tasks[j].output/temp_disks[temp_tasks[j].disk_id].speed,0)
                    task.b=now_time+spend_time
                    spend_time+=round(task.size/machine.power,0)
                    task.c=now_time+spend_time
                    spend_time+=round(task.output/disk.speed,0)
                    task.d=now_time+spend_time
                    machine.time=task.d
                    
                    disk.quota-=task.output
                    finish_time=max(finish_time,task.d)
                    #print(task.id,task.a,task.machine_id+1,task.disk_id+1)
        for task in temp_tasks:
            print(task.id,int(task.a),task.machine_id+1,task.disk_id+1)
        # print("finish_time=",finish_time)
        # self.calculate_cost(solution)
        # print("cost=",solution.cost)
        # print("error=",solution.error)
           
        
        

        
            


                
def main():
    start_time=time.time()

    # record_cost=[]
    # record_error=[]
    tasks, machines, disks = load_data()
    global count1  # Declare count1 as global
    global count2 
    count1=0
    count2=0
    


    #for i in tasks:
        #print(i.id,i.size,i.output,i.num_machines,i.affinitive_machines)
        #print(i.env_dep,i.data_dep)

    ga_slover=SA(tasks,machines,disks,pop_size=5)
    ga_slover.solver() 
    ga_slover.display()
    end_time=time.time() 
    # print("count1=",count1)
    # print("count2=",count2)
    # print("time=",end_time-start_time)
    # print("cost=",ga_slover.record_cost)
    # print("error=",ga_slover.record_error)
    # print(count1,count2)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # ax1.plot(range(len(ga_slover.record_cost)), ga_slover.record_cost, label='Cost')
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Cost')
    # ax1.legend()

    # ax2.plot(range(len(ga_slover.record_error)), ga_slover.record_error, label='Error')
    # ax2.set_xlabel('Iteration')
    # ax2.set_ylabel('Error')
    # ax2.legend()

    # plt.tight_layout()
    # plt.show()
if __name__ == "__main__":
    main()