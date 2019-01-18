import numpy as np
import random
from pathlib import Path
import pandas as pd
from PIL import Image
import csv

import torch
from torch.utils import data

from torchvision import transforms
from torchvision.transforms import functional as F

class WhalePredictDataset(data.Dataset):
    def __init__(self,
            images,
            transform=None):

        self.files = list(images)
        self.transform=transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)
        M = np.eye(3)

        if self.transform:
            # Each image transform updates transform matrix `M`
            img, M = self.transform((img, M))
        # For prediction return the inverse transform 
        return img, np.linalg.inv(M), path.name


class WhaleLocalizeDataset(data.Dataset):
    def __init__(self, 
            image_folder, 
            points_file, 
            transform=None, 
            target_transform=None):

        self.image_folder = Path(image_folder)

        def parse_line(line):
            return line[0], tuple(map(float, line[1:]))

        with open(points_file, 'r') as f:
            # data is of form ('sa38fd.jpg', (x1,y1,x2,y2,...xn,yn))
            self.data = tuple(map(parse_line, csv.reader(f)))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, points = self.data[idx]
        points = np.array([points[::2],points[1::2]])

        img = Image.open(self.image_folder / filename)
        M = np.eye(3)

        if self.transform:
            # Each image transform updates transform matrix `M`
            img, M = self.transform((img, M))
            # Add third dimension for affine transform
            points = np.pad(points, [(0,1),(0,0)], 
                    mode='constant', constant_values=1)
            points = (M @ points)[:2,:] # Apply transform to points
        if self.target_transform:
            points = self.target_transform(points)
        return img, points

if __name__ == "__main__":
    pass
