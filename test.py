# 读取输入
l = int(input())  # 任务数量
tasks = []
for i in range(l):
    task_info = list(map(int, input().split()))
    tasks.append(task_info)

n = int(input())  # 机器数量
machines = []
for i in range(n):
    machine_info = list(map(int, input().split()))
    machines.append(machine_info)

m = int(input())  # 磁盘数量
disks = []
for i in range(m):
    disk_info = list(map(int, input().split()))
    disks.append(disk_info)

# 读取数据依赖关系
N = int(input())
data_dependencies = []
for i in range(N):
    i, j = map(int, input().split())
    data_dependencies.append((i, j))

# 读取环境依赖关系
M = int(input())
env_dependencies = []
for i in range(M):
    i, j = map(int, input().split())
    env_dependencies.append((i, j))

# 根据数据依赖关系排序任务
tasks.sort(key=lambda x: x[0])  # 根据任务ID排序，确保顺序一致

# 初始化变量
current_time = [0] * n  # 每台机器的当前可用时间
disk_usage = [0] * m  # 每个磁盘的当前数据使用量
schedule = []  # 记录任务调度详情

# 计算 ceil(a / b)
def ceil_div(a, b):
    return (a + b - 1) // b

# 处理每个任务
for task in tasks:
    task_id, task_size, output_size, k, *affinitive_machines = task
    affinitive_machines = set(affinitive_machines)

    # 计算开始时间 (a_i)
    start_time = 0
    for j in range(l):
        if (j + 1, task_id) in data_dependencies:
            dep_task_id = j + 1
            for idx, scheduled_task in enumerate(schedule):
                if scheduled_task[0] == dep_task_id:
                    start_time = max(start_time, scheduled_task[3])  # c_j
                    break
    
    # 选择合适的机器 (y_i)
    machine_id = -1
    for machine_info in machines:
        if machine_info[0] in affinitive_machines:
            machine_id = machine_info[0]
            break
    
    # 选择合适的磁盘 (z_i)
    disk_id = -1
    for j in range(m):
        if disk_usage[j] + output_size <= disks[j][2]:  # 检查磁盘配额
            disk_id = disks[j][0]
            disk_usage[j] += output_size
            break

    # 计算各阶段时间 (b_i, c_i, d_i)
    read_phase_time = 0
    for j in range(l):
        if (task_id, j + 1) in data_dependencies:
            dep_task_id = j + 1
            for idx, scheduled_task in enumerate(schedule):
                if scheduled_task[0] == dep_task_id:
                    read_phase_time += ceil_div(scheduled_task[2], disks[disk_id - 1][1])
                    break
    
    # 计算执行阶段时间 (c_i)
    execution_phase_time = ceil_div(task_size, machines[machine_id - 1][1])
    
    # 计算完成时间 (d_i)
    finish_time = start_time + read_phase_time + execution_phase_time
    
    # 更新机器当前时间
    current_time[machine_id - 1] = finish_time
    
    # 记录任务调度详情
    schedule.append((task_id, start_time, machine_id, disk_id))

# 输出调度结果
for task_info in schedule:
    print(task_info[0], task_info[1], task_info[2], task_info[3])






class TaskGraph:
    def __init__(self, tasks):
        self.num_tasks = len(tasks)
        self.tasks = tasks
        self.adj_list = [[] for _ in range(self.num_tasks)]
        self.in_degree = [0] * self.num_tasks
        
        # Build the adjacency list and in-degree array
        for idx, task in enumerate(self.tasks):
            for dep_task_id in task.env_dep:
                self.adj_list[dep_task_id].append(idx)
                self.in_degree[idx] += 1
            for dep_task_id in task.data_dep:
                if dep_task_id not in task.env_dep:  # 避免重复计算入度
                    self.adj_list[dep_task_id].append(idx)
                    self.in_degree[idx] += 1
    
    def topological_sort(self):
        # Kahn's algorithm to find topological order
        result_order = []
        
        # Find all tasks with zero in-degree to start with
        zero_in_degree_tasks = [idx for idx in range(self.num_tasks) if self.in_degree[idx] == 0]
        
        while zero_in_degree_tasks:
            # Randomly select a task with zero in-degree
            task_idx_to_remove = random.choice(zero_in_degree_tasks)
            result_order.append(task_idx_to_remove)
            
            # Remove the task and update adjacency list and in-degree
            for neighbor_idx in self.adj_list[task_idx_to_remove]:
                self.in_degree[neighbor_idx] -= 1
                if self.in_degree[neighbor_idx] == 0:
                    zero_in_degree_tasks.append(neighbor_idx)
            
            # Remove the task from zero in-degree list
            zero_in_degree_tasks.remove(task_idx_to_remove)
        
        # Check if all tasks are included in the result order
        if len(result_order) != self.num_tasks:
            return None  # Not all tasks were included, cyclic dependency exists
        
        return result_order

# Example usage
if __name__ == "__main__":
    # Example tasks setup
    tasks = [
        Task(0, 40, 6, 2, [1, 2]),
        Task(1, 20, 6, 2, [1, 2]),
        Task(2, 96, 10, 2, [1, 2]),
        Task(3, 20, 6, 2, [1, 2]),
        Task(4, 60, 0, 2, [1, 2]),
        Task(5, 20, 0, 1, [1])
    ]
    
    # Example data dependencies and environment dependencies (mock data)
    tasks[0].add_data_dependency(1)
    tasks[0].add_env_dependency(3)
    
    tasks[1].add_data_dependency(2)
    tasks[1].add_env_dependency(4)
    
    tasks[2].add_data_dependency(3)
    
    tasks[3].add_data_dependency(1)
    tasks[3].add_env_dependency(4)
    
    tasks[4].add_data_dependency(2)
    
    tasks[5].add_env_dependency(5)
    
    # Create a task graph
    task_graph = TaskGraph(tasks)
    
    # Perform topological sorting
    result_order = task_graph.topological_sort()
    
    if result_order is not None:
        print("Task execution order after topological sorting:")
        for task_idx in result_order:
            print(tasks[task_idx])
    else:
        print("Cyclic dependency detected or not all tasks were included.")