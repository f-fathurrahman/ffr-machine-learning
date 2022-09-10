import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Hyper-parameters 
input_size = 28 * 28  # 784
num_classes = 10 # angka 0-9
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(
    root='../DATASET', 
    train=True, 
    transform=transforms.ToTensor(),
    download=False
)
print("train_dataset")
print(train_dataset.data.shape)

test_dataset = torchvision.datasets.MNIST(
    root='../DATASET', 
    train=False, 
    transform=transforms.ToTensor()
)
print("test_dataset")
print(test_dataset.data.shape)

# Data loader (input pipeline), as iterator (bisa digunakan dalam for loop)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

print("parameters")
for name, param in model.named_parameters():
    print(name, " ", param.data)
    print("Shape: ", param.data.shape)

#model.load_state_dict(torch.load("model_trained_03.ckpt"))

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax (activation function) internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    # over over dataset
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size) # dari matriks menjadi vektor

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # update weight
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step/Batch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += int(labels.size(0))
        correct += int((predicted == labels).sum())

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model_logreg_01.ckpt')
