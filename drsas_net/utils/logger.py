from __future__ import annotations
import time

class SimpleLogger:
    def __init__(self):
        self.start = time.time()

    def log(self, step: int, **metrics):
        elapsed = time.time() - self.start
        items = " ".join(f"{k}={v:.4f}" if isinstance(v,(int,float)) else f"{k}={v}" for k,v in metrics.items())
        print(f"[{elapsed:7.1f}s] step={step} {items}")
