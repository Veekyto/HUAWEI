import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
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
        self.time=0

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
        tasks[y-1].env_dep.add(x-1)
        
    b=int(input())
    for i in range(b):
        x,y=list(map(int, input().split()))
        tasks[y-1].data_dep.add(x-1)
        
        
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

class GA:
    class Individual:
        def __init__(self,num_tasks):
            self.tasks_plan=[-1]*num_tasks
            self.machines_plan=[-1]*num_tasks
            self.disks_plan=[-1]*num_tasks
            self.cost=0
            self.error=0

    def __init__(self, tasks, machines, disks,  pop_size=50, crossover_rate=0.8, mutation_rate=0.2):
        self.tasks = tasks
        self.machines = machines
        self.disks = disks
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_tasks = len(tasks)
        self.num_machines = len(machines)
        self.num_disks = len(disks)
        self.population = self.initialize_population()
        
        for individual in self.population:
            self.calculate_cost(individual)
            print("cost=",individual.cost,"error=",individual.error)
    
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            sort_task=TaskGraph(self.tasks)
            disk_empty = [disk.quota for disk in self.disks]
            individual = GA.Individual(self.num_tasks)
            individual.tasks_plan = sort_task.topological_sort()
            for i in range(self.num_tasks):
                task=self.tasks[individual.tasks_plan[i]]
                individual.machines_plan[i] = random.choice(task.affinitive_machines)
                chosen_disk=random.randint(0,self.num_disks-1)
                while task.output > disk_empty[chosen_disk]:
                    chosen_disk = random.randint(0, self.num_disks - 1)
                    print("task.output=",task.output,"chosen_disk=",chosen_disk,"disk_empty[chosen_disk]",disk_empty[chosen_disk])
                individual.disks_plan[i] = chosen_disk
                disk_empty[chosen_disk] -= task.output
            population.append(individual)
            if _==1:
                print("tasks_plan=",individual.tasks_plan)
                print("machines_plan",individual.machines_plan)
                print("disk_plan=",individual.disks_plan)
        return population
    
    def select_parents(self):
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        return min(tournament,key=lambda x: x.cost)

    def crossover(self, parent1, parent2):
        offspring=copy.deepcopy(parent1)
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.num_tasks - 1)
            offspring.tasks_plan=parent1.tasks_plan[:crossover_point] + parent2.tasks_plan[crossover_point:]
            offspring.machines_plan=parent1.machines_plan[:crossover_point] + parent2.machines_plan[crossover_point:]
            if random.random() < 0.5:
                offspring.disks_plan=parent1.disks_plan[:crossover_point] + parent2.disks_plan[crossover_point:]
        return offspring

    def mutate(self, solution):
        if random.random() < self.mutation_rate:
            if random.random() < 0.5:
                mutation_point1, mutation_point2 = random.sample(range(self.num_tasks), 2)
                solution.tasks_plan[mutation_point1], solution.tasks_plan[mutation_point2] = solution.tasks_plan[mutation_point2], solution.tasks_plan[mutation_point1]
                solution.machines_plan[mutation_point1], solution.machines_plan[mutation_point2] = solution.machines_plan[mutation_point2], solution.machines_plan[mutation_point1]
            if random.random() < 0.5:
                mutation_point1, mutation_point2 = random.sample(range(self.num_tasks), 2)
                solution.disks_plan[mutation_point1], solution.disks_plan[mutation_point2] = solution.disks_plan[mutation_point2], solution.disks_plan[mutation_point1]
            if random.random() < 0.5:
                mutation_point = random.randint(0, self.num_tasks - 1)
                solution.machines_plan[mutation_point] = random.choice(self.tasks[solution.tasks_plan[mutation_point]].affinitive_machines)
        return solution
    
    def calculate_cost(self,solution):
        finish_time=0
        count=0
        temp_tasks=copy.deepcopy(self.tasks)
        temp_machines=copy.deepcopy(self.machines)
        temp_disks=copy.deepcopy(self.disks)
        for i in range(self.num_tasks):
            task=temp_tasks[solution.tasks_plan[i]]
            machine = temp_machines[solution.machines_plan[i]]
            disk=temp_disks[solution.disks_plan[i]]
            now_time=max(task.a,machine.time,disk.time)
            task.a=now_time
            spend_time=0
            for j in task.data_dep:
                spend_time+=round(temp_tasks[j-1].output/disk.speed,0)
            task.b=now_time+spend_time
            spend_time+=round(task.size/machine.power,0)
            task.c=now_time+spend_time
            spend_time+=round(task.output/disk.speed,0)
            task.d=now_time+spend_time
            machine.time=task.d
            disk.time=task.d
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
                    env_error+=1
            for j in i.data_dep:
                if i.a<tasks[j].d:
                    data_error+=1
        for i in disks:
            if i.quota<0:
                quota_error+=abs(i.quota)
        return env_error*3+round(data_error*3,0)+quota_error*2
    
    def solver(self,max_iteration=200):
        for i in range(max_iteration):
            for solution in self.population:
                self.calculate_cost(solution)
            new_population=[]
            it=0
            while it<self.pop_size:
                parent1=self.select_parents()
                parent2=self.select_parents()
                child=self.crossover(parent1,parent2)
                child=self.mutate(child)
                self.calculate_cost(child)
                if child.error==0:
                    new_population.append(child)
                    it+=1
                
            self.population=new_population
            self.population.sort(key=lambda x: x.cost)
            record_cost.append(self.population[0].cost)
            record_error.append(self.population[0].error)
#display
        temp_tasks=copy.deepcopy(self.tasks)
        temp_machines=copy.deepcopy(self.machines)
        temp_disks=copy.deepcopy(self.disks)
        solution=self.population[0]
        finish_time=0
        for i in range(self.num_tasks):
            task=temp_tasks[solution.tasks_plan[i]]
            machine = temp_machines[solution.machines_plan[i]]
            task.machine_id=solution.machines_plan[i]
            disk=temp_disks[solution.disks_plan[i]]
            task.disk_id=solution.disks_plan[i]
            now_time=max(task.a,machine.time,disk.time)
            task.a=now_time
            spend_time=0
            for j in task.data_dep:
                spend_time+=round(temp_tasks[j-1].output/disk.speed,0)
            task.b=now_time+spend_time
            spend_time+=round(task.size/machine.power,0)
            task.c=now_time+spend_time
            spend_time+=round(task.output/disk.speed,0)
            task.d=now_time+spend_time
            machine.time=task.d
            disk.time=task.d
            disk.quota-=task.output
            finish_time=max(finish_time,task.d)
            print(task.id,task.a,task.machine_id,task.disk_id)
        print("finish_time=",finish_time)
        
        

        
            


                


record_cost=[]
record_error=[]
tasks, machines, disks = load_data()
count=0

for i in tasks:
    print(i.id,i.size,i.output,i.num_machines,i.affinitive_machines)
    print(i.env_dep,i.data_dep)

ga_slover=GA(tasks,machines,disks,pop_size=50)
ga_slover.solver()  
print("cost=",record_cost)
print("error=",record_error)
print(count)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(range(len(record_cost)), record_cost, label='Cost')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost')
ax1.legend()

ax2.plot(range(len(record_error)), record_error, label='Error')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Error')
ax2.legend()

plt.tight_layout()
plt.show()