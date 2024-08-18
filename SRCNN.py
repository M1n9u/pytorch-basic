from glob import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import random

filenames = glob("data/T91/*.png")
original_images = []

for filename in filenames:
    if os.path.exists(filename):
        img = cv2.imread(filename)
        if img is not None:
            original_images.append(img)

scale = 1.5
hr_images = []
for img in iter(original_images):
    height, width = img.shape[:2]
    hmin, wmin = round(height/scale), round(width/scale)
    hmin, wmin = round(hmin*scale), round(wmin*scale)
    h_center, w_center = height//2, width//2
    h_start, w_start = h_center - (hmin//2), w_center - (wmin//2)
    crop_img = np.array(img[h_start:h_start + hmin, w_start:w_start + wmin, :]).astype(np.float32)
    hr_images.append(crop_img/255.0)

del original_images

lr_patch = []
hr_patch = []
lr_images = []
for hr in iter(hr_images):
    f = 1.0/float(scale)
    downsample = cv2.resize(hr, (0,0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
    lr = cv2.resize(downsample, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lr_images.append(lr)
    patch_size = 33
    stride = 14
    output_size = 33
    pad = (patch_size - output_size) // 2
    for i in range(0, lr.shape[0] - patch_size + 1, stride):
        for j in range(0, lr.shape[1] - patch_size + 1, stride):
            lr_patch.append(lr[i:i + patch_size, j:j + patch_size])
            hr_patch.append(hr[i + pad:i + pad + output_size, j + pad:j + pad + output_size])
lr_patches, hr_patches = np.array(lr_patch), np.array(hr_patch)
del lr_patch, hr_patch
gc.collect()
print(lr_patches.shape)

idx = random.randint(0, len(lr_patches) - 1)

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow(lr_patches[idx])
plt.title("Low Resolution")

plt.subplot(1, 2, 2)
plt.imshow(hr_patches[idx])
plt.title("High Resolution")
plt.show()

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

# SRCNN 모델
class srcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, 1)
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x

def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

model = srcnn()
model.apply(initialize_weights)

summary(model,(3,33,33))

# 데이터셋
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transform

class customDataset(Dataset):
    def __init__(self, image_data, labels):
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return (len(self.image_data))

    def __getitem__(self, index):
        image = self.image_data[index]
        label = self.labels[index]
        trans = transform.ToTensor()
        return trans(image), trans(label)

train_dataset = customDataset(lr_patches, hr_patches)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 학습
from torch.optim import Adam
from tqdm import tqdm
import math

def psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0: # label과 output이 완전히 일치하는 경우
        return 100
    else:
        psnr = 20 * math.log10(max_val/rmse)
        return psnr

epochs = 25
criterion = nn.MSELoss(reduction='sum')
optimizer = Adam([
    {"params": model.conv1.parameters(), "lr" : 1e-3},
    {"params": model.conv2.parameters(), "lr" : 1e-3},
    {"params": model.conv3.parameters(), "lr" : 1e-4}], betas=(0.9, 0.999))

model.train()
train_psnr = []
for epochs in tqdm(range(epochs)):
    running_loss = 0.0
    count = 0
    for i, (lowres, highres) in enumerate(train_loader):
        output = model(lowres)
        loss = criterion(output, highres)
        running_loss += psnr(highres, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += len(output)
    running_loss /= count
    train_psnr.append(running_loss)


plt.figure()
plt.plot(range(1,epochs+2),train_psnr)
plt.title("Train PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.show()

# 결과 출력

def denormalize(img):
    img *= 255.0
    img = np.clip(img, 0, 255)
    return np.uint8(img)

model.eval()
idx = random.randint(0, len(lr_images) - 1)
low, high = torch.from_numpy(lr_images[idx]).unsqueeze(0), torch.from_numpy(hr_images[idx]).unsqueeze(0)
low, high = low.permute(0, 3, 1, 2), high.permute(0, 3, 1, 2)

with torch.no_grad():
    recon = model(low)

low_numpy = denormalize(low[0].permute(1, 2, 0).cpu().numpy())
recon_numpy = denormalize(recon[0].permute(1, 2, 0).cpu().numpy())
high_numpy = denormalize(high[0].permute(1, 2, 0).cpu().numpy())

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(low_numpy)
plt.title("Low Resolution")

plt.subplot(1, 3, 2)
plt.imshow(high_numpy)
plt.title("High Resolution")

plt.subplot(1, 3, 3)
plt.imshow(recon_numpy)
plt.title("SRCNN Reconstructed")

plt.show()