from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
import matplotlib.pyplot as plt

# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
# Batch size = 64
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Plot images from each category
data, target = next(iter(train_loader))
_, class_idx = np.unique(target, return_index=True)
class_name = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
plt.figure(figsize=(10,4))
for i, idx in enumerate(class_idx):
    plt.subplot(2,5,i+1)
    plt.imshow(data[idx].permute(1,2,0).detach().numpy())
    plt.title(class_name[i])
plt.tight_layout()
plt.show()

# Model
class cnn_Model(nn.Module):
    def __init__(self):
        super(cnn_Model, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU())
        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU())
        self.linear1 = nn.Sequential(nn.Linear(in_features=8*8*32, out_features=1024),
                                     nn.ReLU())
        self.classifier = nn.Linear(in_features=1024, out_features=10)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(self.cnn1(x))
        x = self.maxpool(self.cnn2(x))
        x = x.view(-1,8*8*32)
        x = self.linear1(x)
        x = self.classifier(x)
        return x

model = cnn_Model()
summary(model, input_size=(3, 32, 32))

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# Train
num_epoch = 20
train_loss = []
train_acc = []
model.train()
print("Train start")
for epoch in range(num_epoch):
    running_loss = 0.0
    running_acc = 0.0
    count = 0
    for (data, target) in train_loader:
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_acc += (torch.argmax(output, dim=1) == target).sum().detach().numpy()
        count += len(data)
    running_loss /= len(train_loader)
    running_acc = float(running_acc) / float(count) * 100.0
    train_loss.append(running_loss)
    train_acc.append(running_acc)
    print("Epoch: {}, Loss: {:0.2f}, Train accuracy: {:0.2f} %".format(epoch+1, running_loss, running_acc))

plt.figure()
plt.subplot(1,2,1)
plt.plot(range(len(train_loss)), train_loss)
plt.title("Train Loss")
plt.subplot(1,2,2)
plt.plot(range(len(train_acc)),train_acc)
plt.title("Train Accuracy")
plt.show()

del running_loss, running_acc

# Test
model.eval()

test_accuracy = 0.0
count = 0
for (data, target) in test_loader:
    output = model(data)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_accuracy += (torch.argmax(output, dim=1) == target).sum().detach().numpy()
    count += len(data)
test_accuracy = float(test_accuracy) / float(count) * 100.0

print("Test accuracy: {:0.2f} %".format(test_accuracy))