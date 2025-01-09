import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

def print_system_memory_info():
    # RAM usage
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / 1e9:.2f} GB")
    print(f"Available RAM: {mem.available / 1e9:.2f} GB")
    print(f"Used RAM: {mem.used / 1e9:.2f} GB")
    print(f"RAM Usage: {mem.percent}%\n")
    
    # Disk usage
    try:
        disk = psutil.disk_usage('C:\\')  # Windows için geçerli bir sürücü yolu
        print(f"Total Disk Space: {disk.total / 1e9:.2f} GB")
        print(f"Used Disk Space: {disk.used / 1e9:.2f} GB")
        print(f"Free Disk Space: {disk.free / 1e9:.2f} GB")
        print(f"Disk Usage: {disk.percent}%\n")
    except Exception as e:
        print("An error occurred while retrieving disk usage information:")
        print(e)

    # GPU memory usage (just NVIDIA GPU)
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # first select GPU
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"Total GPU Memory: {info.total / 1e9:.2f} GB")
        print(f"Used GPU Memory: {info.used / 1e9:.2f} GB")
        print(f"Free GPU Memory: {info.free / 1e9:.2f} GB")
        nvmlShutdown()
    except Exception as e:
        print(f"GPU memory information could not be retrieved: {e}")
#--------------------------------------------------------------------
import gc
import os
import psutil

def clear_memory():
    #Invoke garbage collection manually
    gc.collect()
    print("Cleared memory.")

    # Check system RAM usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)
    print(f"now memory used: {memory_usage:.2f} MB")

