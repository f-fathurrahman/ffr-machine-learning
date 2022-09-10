import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

X, y = datasets.make_moons(n_samples=200,
    shuffle=True,
    noise=0.1,
    random_state=RANDOM_SEED
)

colors = ["blue", "red"]
plt.clf()
idx1 = (y == 0)
plt.scatter(X[idx1, 0], X[idx1, 1], color="blue", marker="*")
idx2 = (y == 1)
plt.scatter(X[idx2, 0], X[idx2, 1], color="red", marker="o")
plt.savefig("IMG_class_01.pdf")

# Convert data to tensor
inputs = torch.tensor(X, dtype=torch.float)
targets = torch.tensor(y)

#targets = torch.nn.functional.one_hot(targets)
# to get the target containing class indices back,
# you could use torch.argmax(one_hot, dim=1)

# Hyper-parameters
input_size = 2
output_size = 2 # no of class
Nhidden = 5

model = nn.Sequential(
    nn.Linear(input_size,output_size),
    nn.LogSoftmax(dim=1)
)

num_epochs = 500
learning_rate = 0.1

# Loss and optimizer
criterion = nn.NLLLoss() # negative log likelihood
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # stochastic gradient descent

# Train the model
for epoch in range(num_epochs):

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()   # calculate loss function derivative
    optimizer.step()  # will update the model parameters
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


print("parameters")
for name, param in model.named_parameters():
    print(name, " ", param.data)

# Plot the graph
with torch.no_grad():
    y = model(inputs).detach().numpy() # back to numpy
    print(y.shape)
    print(y)
    #plt.clf()
    #idx1 = (y == 0)
    #plt.scatter(X[idx1, 0], X[idx1, 1], color="blue", marker="*")
    #idx2 = (y == 1)
    #plt.scatter(X[idx2, 0], X[idx2, 1], color="red", marker="o")
    #plt.legend()
    #plt.savefig("IMG_class_01.pdf")

# Save the model checkpoint
torch.save(model.state_dict(), 'model_regr_seq_02.ckpt')