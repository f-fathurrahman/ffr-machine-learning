import torch

x = torch.tensor([1.1], requires_grad=True)
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

print("This is cos(x): (shoule be the same as x.grad)")
print(2*torch.cos(x))