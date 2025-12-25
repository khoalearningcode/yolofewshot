#!/usr/bin/env python3

import torch
import gc
import time

print("🧹 Clearing GPU memory...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"✅ GPU memory cleared. Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f}GB")
else:
    print("⚠️  CUDA not available")

gc.collect()
time.sleep(1)
print("✅ Done")
