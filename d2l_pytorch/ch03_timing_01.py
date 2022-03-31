import math
import numpy as np
import torch

#from d2l import torch as d2l

from my_timer import MyTimer

N = 10000
a = torch.ones(N)
b = torch.ones(N)
c = torch.zeros(N)
timer = MyTimer()
for i in range(N):
    c[i] = a[i] + b[i]

print(f"Using loop: {timer.stop():.5f} sec")

timer.start()
d = a + b
print(f"Using + directly: {timer.stop():.5f} sec")