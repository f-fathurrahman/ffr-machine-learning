# Example of detach

import torch

x = torch.arange(4.0, requires_grad=True, dtype=torch.float64)
print("x = ")
print(x)

# Some calculation
y = x*x
print("y = ")
print(y)

u = y.detach()
z = u*x

# Calculate gradient
z.sum().backward()

print("x.grad = ", x.grad)
print("u = ", u) # should be the same as x.grad

# Since the computation of y was recorded, we can subsequently invoke backpropagation on y to
# get the derivative of y = x * x with respect to x, which is 2 * x.
x.grad.zero_()
y.sum().backward()
print("x.grad = ", x.grad)
print("2*x = ", 2*x)

