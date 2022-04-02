import torch

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
print(x.grad)

# Some calculation
y = 2*torch.dot(x, x)
print(y)
# Backward pass, calculate the gradient
y.backward()
print(x.grad)
