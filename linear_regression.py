import seaborn as sns
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

df = sns.load_dataset('titanic')
del df['alive']
print(df.head(3))
input_cols = df.columns[1:]
target_cols = df.columns[0]
categorical_cols = df.select_dtypes('object').columns


def data_to_numpy(frame):
    dataframe = frame.copy(deep=True)
    for col in categorical_cols:
        dataframe[col] = dataframe[col].astype('category').cat.codes
    data = dataframe[input_cols].to_numpy()
    target = dataframe[target_cols].to_numpy()
    return data, target


X, Y = data_to_numpy(df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
print('train dataset shape: ', X_train.shape, Y_train.shape)
print('test dataset shape: ', X_test.shape, Y_test.shape)

class linear_Regression(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(linear_Regression,self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)

    def forward(self,x):
        out = self.linear(x)
        return out


X_train_torch = torch.from_numpy(X_train).type(torch.FloatTensor)
Y_train_torch = torch.from_numpy(Y_train).type(torch.FloatTensor)
X_test_torch = torch.from_numpy(X_test).type(torch.FloatTensor)
Y_test_torch = torch.from_numpy(Y_test).type(torch.FloatTensor)

train_dataset = TensorDataset(X_train_torch, Y_train_torch)
test_dataset = TensorDataset(X_test_torch, Y_test_torch)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Train
linear_model = linear_Regression(13, 1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear_model.parameters, lr=0.01, weight_decay=0.001)
linear_model.train()
for epoch in range(20):
    running_loss = 0.0
    for (data, target) in train_loader:
        output = linear_model(data)
        loss = criterion(output, target)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {}, loss: {}".format(epoch+1, running_loss/len(train_loader)))

# Test
linear_model.eval()
accuracy = 0.0
count = 0
for (data, target) in test_loader:
    output = linear_model(data).item()
    pred = np.array(list(map(lambda x: x > 0.5, output)))
    corr = target.type(torch.int64).numpy()
    accuracy += (pred == corr).sum()
    count += len(corr)

print("Test accuracy: {} %".format(accuracy/float(count)))