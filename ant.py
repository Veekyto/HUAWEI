
import math
import random
import copy

class Task:
    def __init__(self, id, size, data_size, affinities):
        """
        Task类表示一个任务，包括任务的基本属性和相关依赖。
        
        Args:
        - id: 任务ID
        - size: 任务的执行时间（单位：秒）
        - data_size: 任务输出数据的大小
        - affinities: 任务可执行的机器列表
        
        Attributes:
        - id: 任务ID
        - size: 任务的执行时间（单位：秒）
        - data_size: 任务输出数据的大小
        - affinities: 任务可执行的机器列表
        - dependencies: 数据依赖的任务集合
        - dependents: 环境依赖的任务集合
        - earliest_start_time: 任务最早开始时间
        - start_time: 任务实际开始时间
        - machine_id: 分配的机器ID
        - disk_id: 存储数据的磁盘ID
        """
        self.id = id
        self.size = size
        self.data_size = data_size
        self.affinities = affinities
        self.dependencies = set()  # 数据依赖的任务集合
        self.dependents = set()    # 环境依赖的任务集合
        self.earliest_start_time = 0
        self.start_time = 0
        self.machine_id = -1
        self.disk_id = -1

class Machine:
    def __init__(self, id, power):
        """
        Machine类表示一个机器，包括机器的基本属性。
        
        Args:
        - id: 机器ID
        - power: 机器的处理能力
        
        Attributes:
        - id: 机器ID
        - power: 机器的处理能力
        """
        self.id = id
        self.power = power

class Disk:
    def __init__(self, id, speed, quota):
        """
        Disk类表示一个磁盘，包括磁盘的基本属性。
        
        Args:
        - id: 磁盘ID
        - speed: 磁盘的读写速度
        - quota: 磁盘的存储配额
        
        Attributes:
        - id: 磁盘ID
        - speed: 磁盘的读写速度
        - quota: 磁盘的存储配额
        - used_quota: 已使用的存储配额
        """
        self.id = id
        self.speed = speed
        self.quota = quota
        self.used_quota = 0

class AntColony:
    def __init__(self, tasks, machines, disks, data_deps, env_deps):
        """
        AntColony类实现了蚂蚁算法来解决任务调度和数据分配问题。
        
        Args:
        - tasks: 任务列表
        - machines: 机器列表
        - disks: 磁盘列表
        - data_deps: 数据依赖关系列表
        - env_deps: 环境依赖关系列表
        
        Attributes:
        - tasks: 任务列表
        - machines: 机器列表
        - disks: 磁盘列表
        - data_deps: 数据依赖关系列表
        - env_deps: 环境依赖关系列表
        - num_tasks: 任务数量
        - num_machines: 机器数量
        - num_disks: 磁盘数量
        - pheromone: 信息素矩阵
        - best_solution: 最优解决方案
        - best_makespan: 最优解的完成时间
        """
        self.tasks = tasks
        self.machines = machines
        self.disks = disks
        self.data_deps = data_deps
        self.env_deps = env_deps
        self.num_tasks = len(tasks)
        self.num_machines = len(machines)
        self.num_disks = len(disks)
        self.pheromone = [[1.0] * self.num_machines for _ in range(self.num_tasks)]
        self.best_solution = None
        self.best_makespan = float('inf')

    def solve(self, max_iterations=100, num_ants=10, alpha=1.0, beta=2.0, evaporation_rate=0.5):
        """
        解决问题的主函数，使用蚂蚁算法进行优化。

        Args:
        - max_iterations: 最大迭代次数
        - num_ants: 蚂蚁数量
        - alpha: 信息素重要程度因子
        - beta: 启发函数因子
        - evaporation_rate: 信息素蒸发率
        
        Returns:
        - best_solution: 最优解决方案
        - best_makespan: 最优解决方案的完成时间
        """
        for _ in range(max_iterations):
            solutions = []

            for ant_id in range(num_ants):
                solution = self.construct_solution(alpha, beta)
                self.local_pheromone_update(solution)
                makespan = self.calculate_makespan(solution)
                solutions.append((solution, makespan))

                if makespan < self.best_makespan:
                    self.best_solution = solution
                    self.best_makespan = makespan

            self.global_pheromone_update(evaporation_rate)

        return self.best_solution, self.best_makespan

    def construct_solution(self, alpha, beta):
        """
        构造一个解决方案，即任务的分配方案。

        Args:
        - alpha: 信息素重要程度因子
        - beta: 启发函数因子
        
        Returns:
        - solution: 当前解决方案
        """
        solution = [-1] * self.num_tasks

        for task_id in range(self.num_tasks):
            allowed_machines = self.tasks[task_id].affinities
            probabilities = [0.0] * self.num_machines
            cumulative_probabilities = [0.0] * self.num_machines
            sum_prob = 0.0

            for machine_id in allowed_machines:
                if self.is_feasible_machine(task_id, machine_id, solution):
                    probabilities[machine_id - 1] = self.pheromone[task_id][machine_id - 1] ** alpha
                    sum_prob += probabilities[machine_id - 1]

            if sum_prob > 0:
                for i in range(self.num_machines):
                    cumulative_probabilities[i] = probabilities[i] / sum_prob
            else:
                cumulative_probabilities = [1.0 / len(allowed_machines)] * len(allowed_machines)

            selected_machine = self.select_machine(cumulative_probabilities, allowed_machines)
            solution[task_id] = selected_machine

        return solution

    def is_feasible_machine(self, task_id, machine_id, solution):
        """
        判断分配给任务的机器是否合法（满足约束条件）。

        Args:
        - task_id: 任务ID
        - machine_id: 机器ID
        - solution: 当前解决方案
        
        Returns:
        - bool: 是否合法
        """
        task = self.tasks[task_id]
        task_machine = machine_id
        for dependent_task_id in task.dependencies:
            if solution[dependent_task_id] == task_machine:
                return False
        return True

    def select_machine(self, cumulative_probabilities, allowed_machines):
        """
        选择一个机器，根据概率分布选择。

        Args:
        - cumulative_probabilities: 累积概率列表
        - allowed_machines: 可选的机器列表
        
        Returns:
        - selected_machine: 选择的机器ID
        """
        random_value = random.random()
        cumulative_probability = 0.0
        for i, prob in enumerate(cumulative_probabilities):
            cumulative_probability += prob
            if cumulative_probability >= random_value:
                return allowed_machines[i]

    def local_pheromone_update(self, solution):
        """
        更新局部信息素，增加经过的路径信息素浓度。

        Args:
        - solution: 当前解决方案
        """
        for task_id in range(self.num_tasks):
            machine_id = solution[task_id]
            self.pheromone[task_id][machine_id - 1] += 1.0

    def global_pheromone_update(self, evaporation_rate):
        """
        更新全局信息素，使用信息素蒸发率。

        Args:
        - evaporation_rate: 信息素蒸发率
        """
        for task_id in range(self.num_tasks):
            for machine_id in range(self.num_machines):
                self.pheromone[task_id][machine_id] *= evaporation_rate

    def calculate_makespan(self, solution):
        """
        计算解决方案的总完成时间（makespan）。

        Args:
        - solution: 当前解决方案
        
        Returns:
        - makespan: 完成时间
        """
        finish_time = [0] * self.num_tasks
        for task_id in range(self.num_tasks):
            machine_id = solution[task_id]
            task = self.tasks[task_id]
            finish_time[task_id] = task.size + max(finish_time[dep_id] for dep_id in task.dependencies)

        return max(finish_time)

# 示例用法
if __name__ == "__main__":
    # 构造任务、机器和磁盘对象
    tasks = [
        Task(id=0, size=5, data_size=10, affinities=[1, 2]),
        Task(id=1, size=3, data_size=8, affinities=[2, 3]),
        Task(id=2, size=2, data_size=5, affinities=[1, 3]),
        Task(id=3, size=4, data_size=9, affinities=[2])
    ]

    machines = [
        Machine(id=1, power=10),
        Machine(id=2, power=8),
        Machine(id=3, power=12)
    ]

    disks = [
        Disk(id=1, speed=100, quota=1000),
        Disk(id=2, speed=120, quota=1500),
        Disk(id=3, speed=80, quota=800)
    ]

    # 设置任务间的数据依赖关系和环境依赖关系
    tasks[1].dependencies.add(0)
    tasks[2].dependencies.add(0)
    tasks[2].dependencies.add(1)
    tasks[3].dependencies.add(1)

    # 创建AntColony对象并解决问题
    ant_colony = AntColony(tasks, machines, disks, [], [])
    best_solution, best_makespan = ant_colony.solve(max_iterations=100, num_ants=10, alpha=1.0, beta=2.0, evaporation_rate=0.5)

    # 输出结果
    print("Best Solution:", best_solution)
    print("Best Makespan:", best_makespan)


    ##python

class AntColony:
    # 省略其他代码...

    def construct_solution(self, alpha, beta):
        """
        构造一个解决方案，即任务的分配方案。

        Args:
        - alpha: 信息素重要程度因子
        - beta: 启发函数因子
        
        Returns:
        - solution: 当前解决方案
        """
        solution = [-1] * self.num_tasks

        for task_id in range(self.num_tasks):
            allowed_machines = self.tasks[task_id].affinities
            probabilities = [0.0] * self.num_machines
            cumulative_probabilities = [0.0] * self.num_machines
            sum_prob = 0.0

            for machine_id in allowed_machines:
                if self.is_feasible_machine(task_id, machine_id, solution):
                    probabilities[machine_id - 1] = self.pheromone[task_id][machine_id - 1] ** alpha
                    sum_prob += probabilities[machine_id - 1]

            if sum_prob > 0:
                for i in range(self.num_machines):
                    cumulative_probabilities[i] = probabilities[i] / sum_prob
            else:
                cumulative_probabilities = [1.0 / len(allowed_machines)] * len(allowed_machines)

            selected_machine = self.select_machine(cumulative_probabilities, allowed_machines)
            solution[task_id] = selected_machine
            
            # 选择磁盘
            selected_disk = self.select_disk(task_id, solution)
            self.tasks[task_id].disk_id = selected_disk

        return solution

    def select_disk(self, task_id, solution):
        """
        选择一个磁盘，使得任务的输出数据可以存储在该磁盘上。

        Args:
        - task_id: 任务ID
        - solution: 当前解决方案
        
        Returns:
        - disk_id: 选择的磁盘ID
        """
        task = self.tasks[task_id]
        for disk in self.disks:
            if disk.used_quota + task.data_size <= disk.quota:
                disk.used_quota += task.data_size
                return disk.id
        return -1  # 如果没有可用的磁盘，则返回-1或者需要其他处理方式

# 示例用法
if __name__ == "__main__":
    # 构造任务、机器和磁盘对象，设置任务间的依赖关系
    # tasks, machines, disks 的构造略去，直接展示如何使用 AntColony 类

    # 创建 AntColony 对象并解决问题
    ant_colony = AntColony(tasks, machines, disks, [], [])
    best_solution, best_makespan = ant_colony.solve(max_iterations=100, num_ants=10, alpha=1.0, beta=2.0, evaporation_rate=0.5)

    # 输出结果
    print("Best Solution:", best_solution)
    print("Best Makespan:", best_makespan)