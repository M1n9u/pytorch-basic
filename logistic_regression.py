from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

# Iris dataset
X, Y = datasets.load_iris(return_X_y=True)

# Sepal data distribution
plt.figure(figsize=(10, 7))
plt.subplot(1,2,1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette='muted')
plt.title('Sepal', fontsize=17)

# Petal data distribution
plt.subplot(1,2,2)
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=Y, palette='muted')
plt.title('Petal', fontsize=17)
plt.tight_layout()
plt.show()

# Select Petal data
X_numpy, Y_numpy = np.array(X[:, 2:]), np.array(Y)
# Train : Test = 7 : 3
X_train, X_test, Y_train, Y_test = train_test_split(X_numpy, Y_numpy, train_size=0.7, shuffle=True)
X_train_torch, Y_train_torch = torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long()
X_test_torch, Y_test_torch = torch.from_numpy(X_test).float(), torch.from_numpy(Y_test)

# Linear model for logistic regression
class linearModel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(linearModel, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.linear = nn.Linear(self.in_channel, 10)
        self.linear2 = nn.Linear(10, self.out_channel)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        return x


linear_model = linearModel(2,3)
criterion = nn.CrossEntropyLoss() # Softmax + MSELoss
optimizer = Adam(params=linear_model.parameters(), lr=0.02, weight_decay=0.01)
n_epoch = 50

# Train
linear_model.train()
running_loss = []
for epoch in range(n_epoch):
    output = linear_model(X_train_torch)
    loss = criterion(output, Y_train_torch)
    running_loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Running loss plot
plt.figure()
plt.subplot(1,2,1)
plt.plot(range(1, len(running_loss)+1), running_loss)
plt.title("Running Loss")

# Test
linear_model.eval()
pred = torch.argmax(linear_model(X_test_torch), dim=1)
pred_numpy = pred.detach().numpy()
accuracy = (pred == Y_test_torch).sum().item()
print("Test accuracy:", accuracy/float(len(Y_test))*100.0)
plt.subplot(1,2,2)
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=pred_numpy, palette='muted')
plt.scatter(x=X_test[pred_numpy!=Y_test, 0], y=X_test[pred_numpy!=Y_test, 1], c='r', marker='x')
plt.title("Prediction")
plt.tight_layout()
plt.show()