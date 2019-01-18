from collections import OrderedDict
from pathlib import Path
import random
import numpy as np

from multiprocessing import cpu_count
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import models
from torchvision.utils import make_grid

from dataset import WhaleLocalizeDataset
from transforms import Grayscale, RandomAffine, Resize
from transforms import RandomHorizontalFlip, CenterCrop
from transforms import ToTensor, Normalize, PadToSize, ToPILImage

from transforms import ToBBox, NormalizePoints


def center_bbox(bbox):
    """ bbox (x1,y1,x2,y2) to (cx,cy,w,h) """
    with torch.no_grad():
        _min, _max = bbox[:,:2], bbox[:,2:]
        size = _max - _min
        center = _min + size/2
        return torch.cat([center, size], -1)


def uncenter_bbox(bbox):
    """ bbox (cx,cy,w,h) to (x1,y1,x2,y2) """
    with torch.no_grad():
        center, size = bbox[:,:2], bbox[:,2:]
        _min = center - size/2
        _max = _min + size
        return torch.cat([_min, size], -1)

def scale_bbox(bbox, size=(224.,128.)):
    """ Normalize to range 0,1 """
    with torch.no_grad():
        size = torch.Tensor(size*2).to(bbox.device)
        return bbox / size

def unscale_bbox(bbox, size=(224.,128.)):
    """ Unscale from 0,1 to image size """
    with torch.no_grad():
        size = torch.Tensor(size*2).to(bbox.device)
        return bbox * size

def display_bboxes(images, pred_bboxes, target_bboxes):
    with torch.no_grad():
        images = images.cpu()
        pred = unscale_bbox(pred_bboxes.cpu(), (224.,128.)).round().to(torch.int)
        target = unscale_bbox(target_bboxes.cpu(), (224.,128.)).round().to(torch.int)

        color1 = torch.Tensor([1,0,0]).reshape(3,1,1)
        color2 = torch.Tensor([0,1,0]).reshape(3,1,1)
        for i in range(len(images)):
            images[i,...] = draw_box(images[i], pred[i], color1)
            images[i,...] = draw_box(images[i], target[i], color2)
        
        grid = make_grid(images, nrow=4, normalize=True)
        return grid

def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    x1, y1, x2, y2 = box
    image[:, y1:y1 + 2, x1:x2] = color
    image[:, y2:y2 + 2, x1:x2] = color
    image[:, y1:y2, x1:x1 + 2] = color
    image[:, y1:y2, x2:x2 + 2] = color
    return image


def iou(A, B):
    """Intersection over union (IoU) between box A and box B
    
    Args:
        A (Tensor): batch of first box, coordinates (x1, y1, x2, y2)
        B (Tensor): batch of second box, coordinates (x1, y1, x2, y2)
    """
    Amin, Amax = A[:,:2], A[:,2:]
    Bmin, Bmax = B[:,:2], B[:,2:]

    Imin = torch.max(Amin, Bmin)
    Imax = torch.min(Amax, Bmax)

    Isize = (Imax - Imin)
    Iarea = Isize[:,0] * Isize[:,1]
    Iarea[Iarea < 0] = 0

    Asize = Amax - Amin
    Aarea = Asize[:,0] * Asize[:,1]

    Bsize = Bmax - Bmin
    Barea = Bsize[:,0] * Bsize[:,1]

    return Iarea / (Aarea + Barea - Iarea)

def get_test_transform():
    return torchvision.transforms.Compose([
        Grayscale(3),
        Resize((128,224)), #resize_small_dim=False),
        CenterCrop((128,224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        PadToSize((128, 224)),
    ])

def get_train_transform():
    return torchvision.transforms.Compose([
        Grayscale(3),
        RandomAffine(8, shear=(-25,25), scale=(.6, 1.1), 
            translate=(.1,.1), resample=Image.BILINEAR, fillcolor=(128,)*3),
        Resize((128,224)), #resize_small_dim=False),
        CenterCrop((128,224)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        PadToSize((128, 224)),
    ])

def get_loader(
        image_folder, 
        points_file, 
        batch_size=32,
        transform=None,
        target_transform=None,
        shuffle=True):

    ds = WhaleLocalizeDataset(image_folder, points_file, 
            transform, target_transform)
    return DataLoader(ds, batch_size, shuffle=shuffle, 
            num_workers=cpu_count(), pin_memory=True)

def get_loaders(batch_size=32):
    target_transform = torchvision.transforms.Compose([
        NormalizePoints((128,224)),
        ToBBox(),
        lambda points: torch.from_numpy(points)])

    valid_loader = get_loader('data/train', 'data/train.txt',
            batch_size, get_test_transform(), target_transform)

    train_loader = get_loader('data/train', 'data/valid.txt',
            batch_size, get_train_transform(), target_transform)
    return OrderedDict(train=train_loader, valid=valid_loader)


def get_model(pretrained=True, name='resnet34'):
    model = getattr(models, name)(pretrained=pretrained)
    #model.fc = nn.Linear(model.fc.in_features, 4)
    model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    #model.fc = nn.Sequential(nn.Dropout(), nn.Linear(2048, 4))
    model.fc = nn.Linear(2048, 4)
    return model

