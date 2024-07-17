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

cropped_images = []
for img in iter(original_images):
    height, width = img.shape[:2]
    hmin, wmin = hmin - np.mod(hmin, 4), wmin - np.mod(wmin, 4)
    h_center, w_center = height//2, width//2
    h_start, w_start = h_center - (hmin//2), w_center - (wmin//2)
    cropped_images.append(img[h_start:h_start + hmin, w_start:w_start + wmin, :])
images = np.array(cropped_images).astype(np.float32)
images /= 255.0
del original_images, cropped_images
gc.collect()
print(images.shape)

interpolated_images = []
for img in iter(images):
    hhalf, whalf = hmin//2, wmin//2
    half_img = cv2.resize(img, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    interpolated_images.append(cv2.resize(half_img, (0,0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC))
lowres_images = np.array(interpolated_images)
del interpolated_images
gc.collect()

import torch.nn as nn
from torchsummary import summary

# SRCNN 모델
class srcnn(nn.Module):
    def __init__(self):
        super(srcnn, self).__init__()
        self.patch = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=2, padding_mode='replicate')
        self.nonlinear = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=2, padding_mode='replicate')
        self.recon = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2, padding_mode='replicate')
        self.relu = nn.ReLU()
        nn.init.normal_(self.patch.weight.data, 0.0, 0.02)
        nn.init.normal_(self.nonlinear.weight.data, 0.0, 0.02)
        nn.init.normal_(self.recon.weight.data, 0.0, 0.02)

    def forward(self, x):
        out = self.relu(self.patch(x))
        out = self.relu(self.nonlinear(out))
        out = self.recon(out)
        return out

model = srcnn()

summary(model, (3,78,78))

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

train_dataset = customDataset(lowres_images, images, transform=ToTensor())
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
    img = img.astype(np.uint32)
    return img

model.eval()
lowres, highres = next(iter(train_loader))
idx = random.randint(0, len(lowres)-1)
low, high = lowres[idx:idx+1], highres[idx]
recon = model(low).permute(0,2,3,1)[0]
low = low.permute(0,2,3,1)[0]
high = high.permute(1,2,0)
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