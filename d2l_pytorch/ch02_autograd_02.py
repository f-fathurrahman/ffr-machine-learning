import torch

x = torch.arange(4.0, requires_grad=True)
print("x = ")
print(x)

# Some calculation
y = 2*torch.sin(x).sum()
print("y = ")
print(y)

# Backward pass, calculate the gradient
y.backward()
print("x.grad = ")
print(x.grad)
