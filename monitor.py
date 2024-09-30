
import os
import time
import datetime
import logging
import sys
import multiprocessing


def get_date():
    return datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")



def get_gpu_mem_usage():
    global mem_too_high
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_memory = meminfo.used
    total_memory = meminfo.total
    usage = (used_memory / total_memory)
    pynvml.nvmlShutdown()

    # with open("result/meta/gpu_mem_usage.txt", "a") as f:
    #     f.write(f"{get_date()} usage:{usage*100:.1f}% mem:{used_memory/1024**3:.2f}\n")
    logging.info(f"{get_date()} usage:{usage*100:.1f}% mem:{used_memory/1024**3:.2f}")

    if usage > 0.70:
        logging.exception(f"GPU memory usage is too high: {usage*100:.1f}%")
        mem_too_high = True
        pid, memory_usage = find_max_gpu_memory_process()
        if pid:
            print(f"Process with PID {pid} is using the most GPU memory: {memory_usage / 1024**3:.2f} GB")
        else:
            print("No GPU processes found.")
        kill_processes_by_name()
        kill_python_processes()
        # 调用函数找到 GPU 内存使用量最大的进程
        # raise Exception(f"GPU memory usage is too high: {usage*100:.1f}%")
    return usage

def monitor_gpu_usage(interval=5):
    while True:
        get_gpu_mem_usage()
        time.sleep(interval)


def kill_python_processes(name="python.exe"):
    import psutil
    import signal
    # 遍历所有进程
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # 检查进程名称是否为 "python.exe"
            if proc.info['name'] == 'python.exe':
                pid = proc.info['pid']
                print(f"Killing process {pid} ({proc.info['name']})")
                os.kill(pid,9)
                print(f"Process {pid} killed successfully.")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Failed to kill process: {e}")
        # 调用函数打印所有进程

def kill_processes_by_name(substring="amgx"):
    print(f"Killing processes by name containing '{substring}'")
    import psutil
    import signal
    # 遍历所有进程
    for proc in psutil.process_iter(['pid', 'name']):
        # print(f"proc name: {proc.info['name']}, pid: {proc.info['pid']}")
        try:
            # 检查进程名称是否包含指定的子字符串
            if substring.lower() in proc.info['name'].lower():
                pid = proc.info['pid']
                print(f"Killing process {pid} ({proc.info['name']})")
                os.kill(pid, signal.SIGKILL)
                print(f"Process {pid} killed successfully.")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Failed to kill process: {e}")


import pynvml

def find_max_gpu_memory_process():
    # 初始化 NVML
    pynvml.nvmlInit()
    
    # 获取 GPU 设备数量
    device_count = pynvml.nvmlDeviceGetCount()
    
    max_memory_usage = 0
    max_memory_process = None
    
    # 遍历所有 GPU 设备
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        print(f"GPU {i} has {len(gpu_processes)} processes running")
        print(f"GPU {i} has {pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3:.2f} GB used")
        print(f"GPU processes are: {gpu_processes}")
        # 获取每个进程的 GPU 内存使用情况
        for gpu_process in gpu_processes:
            # if gpu_process.usedGpuMemory :
                max_memory_usage = gpu_process.usedGpuMemory
                max_memory_process = gpu_process
                print(f"Process {gpu_process.pid} is using {gpu_process.usedGpuMemory}")
    
    # 关闭 NVML
    pynvml.nvmlShutdown()
    
    if max_memory_process:
        return max_memory_process.pid, max_memory_usage
    else:
        return None, 0




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s",filename=f'result/meta/monitor.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"monitoring pid:{os.getpid()}")
    # 启动 GPU 使用监控线程
    monitor_gpu_usage()
