import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

import os
import glob
from io import open

filenames = glob.glob('data/names/*.txt')
country = []
names = {}
for files in filenames:
    country_name = os.path.splitext(os.path.basename(files))[0]
    country.append(country_name)
    lines = open(files, encoding='utf-8').read().strip().split('\n')
    lines = [unicodeToAscii(line) for line in lines]
    names[country_name] = lines

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

def nameToTensor(name):
    result = torch.zeros(len(name),1,len(all_letters))
    for idx, chr in enumerate(name):
        result[idx][0][all_letters.find(chr)] = 1
    return result

class simpleRNN(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(simpleRNN, self).__init__()
        self.linear = nn.Linear(in_channel, hidden_channel)
        self.hid = nn.Linear(hidden_channel, hidden_channel)
        self.out = nn.Linear(hidden_channel, out_channel)
        self.hidden_channel = hidden_channel

    def forward(self, x, hid):
        hidden = F.tanh(self.linear(x)+self.hid(hid))
        output = self.out(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_channel)

n_input = len(all_letters)
n_hidden = n_input
n_output = len(country)
criterion = nn.CrossEntropyLoss()
model = simpleRNN(n_input, n_hidden, n_output)
optimizer = Adam(model.parameters(), lr=0.0015)

import random
import numpy as np

def randomSample():
    idx = random.randint(0, len(names) - 1)
    selected_names = list(names.values())[idx]
    name = selected_names[random.randint(0,len(selected_names)-1)]
    category = list(names.keys())[idx]
    x = nameToTensor(name)
    y = torch.zeros(1, len(country))
    y[0][country.index(category)] = 1
    return name, category, x, y

num_train = 100000
train_accuracy = []
correct = 0.0
running_loss = 0.0
confusion = np.zeros((n_output, n_output))
for epoch in range(num_train):
    name, category, x, y = randomSample()
    hidden = model.initHidden()
    for i in range(len(x)):
        output, hidden = model(x[i], hidden)
    optimizer.zero_grad()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    pred, label = torch.argmax(output, dim=1), torch.argmax(y, dim=1)
    correct += (pred == label).item()
    confusion[pred.item()][label.item()] += 1
    if ((epoch+1)%10000) == 0:
        average_loss = running_loss/10000.0
        running_loss = 0.0
        train_accuracy.append(correct/100.0)
        correct = 0.0
        check = '✓' if pred == label else '✗ (%s)' % category
        print('%d %d%% %.4f %s / %s %s' % (
        epoch+1, (epoch+1) / num_train * 100, average_loss, name, country[pred], check))


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2)
ax[0].plot(range(len(train_accuracy)), train_accuracy)
ax[0].set_title("Train Accuracy")
ax[0].set_xlabel("Num of names")
ax[0].set_ylabel("Accuracy(%)")

ax[1].matshow(confusion)
ax[1].set_title("Confusion Map")
ax[1].set_xlabel("Actual class")
ax[1].set_ylabel("Predicted class")
plt.show()