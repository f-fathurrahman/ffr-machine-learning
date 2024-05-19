# From:
# https://kitchingroup.cheme.cmu.edu/blog/2024/05/05/Kolmogorov-Arnold-Networks-KANs-and-Lennard-Jones/

import matplotlib.pyplot as plt
import torch
from my_kan import create_dataset, KAN

def LJ_potential(r):
    r6 = r**6
    return 1 / r6**2 - 1 / r6

dataset = create_dataset(LJ_potential, n_var=1, ranges=[0.95, 2.0],
                         train_num=50)

plt.clf()
plt.plot(dataset['train_input'], dataset['train_label'], 'b.')
plt.xlabel('r')
plt.ylabel('E')
plt.savefig("IMG_dataset.png", dpi=150)

plt.clf()
model = KAN(width=[1, 2, 1])
model.train(dataset, opt="LBFGS", steps=20)
model.plot()
plt.savefig("IMG_train_01.png", dpi=150)


X = torch.linspace(dataset['train_input'].min(),
                   dataset['train_input'].max(), 100)[:, None]

plt.clf()
plt.plot(dataset['train_input'], dataset['train_label'], 'b.', label='data')
plt.plot(X, model(X).detach().numpy(), 'r-', label='fit')
plt.legend()
plt.xlabel('r')
plt.ylabel('E')
plt.savefig("IMG_result_01.png", dpi=150)


X = torch.linspace(0, 5, 1000)[:, None]
plt.clf()
plt.plot(dataset['train_input'], dataset['train_label'], 'b.')
plt.plot(X, model(X).detach().numpy(), 'r-')
plt.savefig("IMG_extrap_01.png", dpi=150)

