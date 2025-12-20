import time
import psutil
import torch

class Metrics:
    def __init__(self, tag=""):
        self.tag = tag
        self.start_time = time.perf_counter()
        self.checkpoints = {}

    def mark(self, name):
        self.checkpoints[name] = time.perf_counter()

    def elapsed(self, name):
        return round(
            (time.perf_counter() - self.checkpoints.get(name, self.start_time)) * 1000, 2
        )

    def total(self):
        return round((time.perf_counter() - self.start_time) * 1000, 2)

    def system(self):
        return {
            "cpu_percent": psutil.cpu_percent(),
            "ram_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
