from glob import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import gc

filenames = glob("data/T91/*.png")
original_images = []

hmin, wmin = 9999, 9999
for filename in filenames:
    if os.path.exists(filename):
        img = cv2.imread(filename)
        if img is not None:
            imsize = img.shape
            hmin, wmin = min(hmin, imsize[0]), min(wmin, imsize[1])
            original_images.append(img)

def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b
    # Cr
    cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b
    return np.uint8(cbcr)

def ycbcr2rgb(im):
    rgb = np.empty_like(im)
    y   = im[:,:,0]
    cb  = im[:,:,1] - 128
    cr  = im[:,:,2] - 128
    # R
    rgb[:,:,0] = y + 1.402 * cr
    # G
    rgb[:,:,1] = y - .34414 * cb - .71414 * cr
    # B
    rgb[:,:,2] = y + 1.772 * cb
    return np.uint8(rgb)

scale = 2
cropped_images = []
for img in iter(original_images):
    height, width = img.shape[:2]
    hmin, wmin = (hmin//scale)*scale, (wmin//scale)*scale
    h_center, w_center = height//2, width//2
    h_start, w_start = h_center - (hmin//2), w_center - (wmin//2)
    crop_img = img[h_start:h_start + hmin, w_start:w_start + wmin, :]
    cropped_images.append(rgb2ycbcr(crop_img))

hr_images = np.array(cropped_images).astype(np.float32)
hr_images /= 255.0

del original_images, cropped_images

lr_patch = []
hr_patch = []
lr_image = []
for hr in iter(hr_images):
    f = 1.0/float(scale)
    downsample = cv2.resize(hr, (0,0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
    lr = cv2.resize(downsample, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lr_image.append(lr)
    patch_size = 33
    stride = 14
    for i in range(0, lr.shape[0] - patch_size + 1, stride):
        for j in range(0, lr.shape[1] - patch_size + 1, stride):
            lr_patch.append(lr[i:i + patch_size, j:j + patch_size])
            hr_patch.append(hr[i:i + patch_size, j:j + patch_size])
lr_patches, hr_patches = np.array(lr_patch), np.array(hr_patch)
lr_images = np.array(lr_image)
del lr_image, lr_patch, hr_patch
gc.collect()
print(lr_patches.shape)

import torch
import torch.nn as nn
from torchsummary import summary

# SRCNN 모델
class srcnn(nn.Module):
    def __init__(self):
        super(srcnn, self).__init__()
        self.patch = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, padding_mode='replicate')
        self.nonlinear = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding_mode='replicate')
        self.recon = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2, padding_mode='replicate')
        self.relu = nn.ReLU()
        nn.init.xavier_normal_(self.patch.weight.data, 0.001)
        nn.init.xavier_normal_(self.nonlinear.weight.data, 0.001)
        nn.init.xavier_normal_(self.recon.weight.data, 0.001)

    def forward(self, x):
        out = self.relu(self.patch(x))
        out = self.relu(self.nonlinear(out))
        out = self.recon(out)
        return out

model = srcnn()

summary(model, (3,33,33))

# 데이터셋
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class customDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.target[idx]
        if self.transform:
            data, target = self.transform(data), self.transform(target)
        return data, target

train_dataset = customDataset(lr_patches, hr_patches, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 학습
from torch.optim import Adam
from tqdm import tqdm

epochs = 100
criterion = nn.MSELoss()
optimizer = Adam([
    {"params": model.patch.parameters(), "lr" : 1e-4},
    {"params": model.nonlinear.parameters(), "lr" : 1e-4},
    {"params": model.recon.parameters(), "lr" : 1e-5}])

model.train()
train_loss = []
for epochs in tqdm(range(epochs)):
    running_loss = 0.0
    count = 0
    for i, (lowres, highres) in enumerate(train_loader):
        output = model(lowres)
        loss = criterion(output, highres)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += len(output)
    running_loss /= count
    train_loss.append(running_loss)

plt.figure()
plt.plot(range(1,epochs+2),train_loss)
plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 결과 출력
import random

def denormalize(img):
    min_px = np.min(img)
    max_px = np.max(img)
    img -= min_px
    img *= 255.0/(max_px-min_px)
    return ycbcr2rgb(img)

model.eval()
idx = random.randint(0, len(lowres)-1)
low, high = torch.from_numpy(lr_images[idx:idx+1]), torch.from_numpy(hr_images[idx])
low = low.permute(0,3,1,2)
recon = model(low).permute(0,2,3,1)[0]
low = low.permute(0,2,3,1)[0]

low_numpy, recon_numpy, high_numpy = low.detach().numpy(), recon.detach().numpy(), high.detach().numpy()
low_numpy, recon_numpy, high_numpy = denormalize(low_numpy), denormalize(recon_numpy), denormalize(high_numpy)


plt.figure()
plt.subplot(1,3,1)
plt.imshow(low_numpy)
plt.title("Low Resolution")
plt.subplot(1,3,2)
plt.imshow(high_numpy)
plt.title("High Resolution")
plt.subplot(1,3,3)
plt.imshow(recon_numpy)
plt.title("SRCNN")
plt.show()