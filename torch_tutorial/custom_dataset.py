#!/usr/bin/env python
#

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

IMG_DIR = '../img/data2/fmnist/test'
LABELS_FILE = IMG_DIR + '/test_labels.csv'
test_dataset = CustomImageDataset(LABELS_FILE, IMG_DIR)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Display images and labels.
test_features, test_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")
for i in range(test_labels.shape[0]):
    img = test_features[i].squeeze()
    label = test_labels[i]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    
