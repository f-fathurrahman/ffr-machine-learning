# Backward for non-scalar variable
import torch

x = torch.arange(4.0, requires_grad=True, dtype=torch.float64)
print("x = ")
print(x)

# Some calculation
y = 2*torch.sin(x) # non-scalar
print("y = ")
print(y)

# Backward pass, calculate the gradient
#y.sum().backward() # or 
y.backward(torch.ones(len(x)))
print("x.grad = ")
print(x.grad)

print("2*cos(x) = ")
print(2*torch.cos(x))
