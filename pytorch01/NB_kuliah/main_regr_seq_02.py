import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def create_dataset(Ndata):
    w_true = [0.25, 1.1]
    noise_var = 1.0
    x_train = -5.0 + 10*np.random.rand(Ndata)
    y_noise = np.sqrt(noise_var)*np.random.randn(Ndata)
    y_train = w_true[1]*x_train + w_true[0] + y_noise
    return x_train, y_train

Ndata = 20
x_train, y_train = create_dataset(Ndata)
plt.clf()
plt.plot(x_train, y_train, linewidth=0, marker="o")
plt.savefig("IMG_regr_02.pdf")

# Convert data to tensor
inputs = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# Hyper-parameters
input_size = 1
output_size = 1
Nhidden = 5

# Linear model + activation, using torch.nn subclass
# Beware accessing global vars
class MyNet(nn.Module):
    # constructor
    def __init__(self, Nhidden_):
        super().__init__()
        self.hidden_linear = nn.Linear(input_size, Nhidden_)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(Nhidden_, output_size)

    def forward(self, inp):
        y1 = self.hidden_linear(inp)
        y2 = self.hidden_activation(y1)
        y3 = self.output_linear(y2)
        return y3


model = MyNet(10)

num_epochs = 500
learning_rate = 0.1

# Loss and optimizer
criterion = nn.MSELoss() # mean square error
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
    predicted = model(inputs).detach().numpy() # back to numpy
    # sort before plotting
    idx = np.argsort(x_train)
    x_train = x_train[idx]
    y_train = y_train[idx]
    predicted = predicted[idx]
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.legend()
    plt.savefig("IMG_regr_seq_02.pdf")

# Save the model checkpoint
torch.save(model.state_dict(), 'model_regr_seq_02.ckpt')