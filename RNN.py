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
import torch.functional as F
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
        return output

    def initHidden(self):
        return torch.zeros(1,self.hidden_channel)

n_input = len(all_letters)
n_hidden = n_input
n_output = len(country)
criterion = nn.CrossEntropyLoss()
model = simpleRNN(n_input, n_hidden, n_output)
optimizer = Adam(model.parameters(), lr=0.005, weight_decay=0.01)

import random

def randomSample():
    idx = random.randint(0, len(names) - 1)
    selected_names = list(names.values())[idx]
    name = selected_names[random.randint(0,len(selected_names)-1)]
    category = list(names.keys())[idx]
    x = nameToTensor(name)
    y = torch.tensor(country.index(category))
    return x, y

