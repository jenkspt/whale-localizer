import copy
import numpy as np
import pandas as pd
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data

from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from utils import get_model, iou, get_loaders, draw_box
from utils import center_bbox, uncenter_bbox, display_bboxes

def train_model(
        model, 
        loaders, 
        optimizer,
        epochs=range(25), 
        device=torch.device('cpu'),
        name='exp1'):

    writers = {
            'train':SummaryWriter(f'logdir/{name}/train'), 
            'valid':SummaryWriter(f'logdir/{name}/valid')
            }
    
    mse = nn.MSELoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0

    for epoch in epochs:
        print(f'Epoch {epoch}/{len(epochs)}\n'+'-'*10)

        # Each epoch has a training and validation phase
        for phase, loader in loaders.items():
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_iou = 0.0

            # Iterate over data.
            for i, (images, bboxes) in enumerate(loader):
                images, bboxes = images.to(device), bboxes.to(torch.float).to(device)
                #bboxC = center_bbox(bboxes)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #predC = torch.sigmoid(model(images))
                    pred_bboxes = torch.sigmoid(model(images))
                    loss = mse(pred_bboxes, bboxes)
                    #pred_bboxes = uncenter_bbox(predC)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if i == 0:
                        grid = display_bboxes(images[:24], 
                                pred_bboxes[:24], bboxes[:24])
                        writers[phase].add_image('Image', grid, epoch)

                # statistics
                running_loss += loss.item() * images.size(0)
                running_iou += iou(pred_bboxes, bboxes).sum()


            epoch_loss = running_loss / len(loader.dataset)
            epoch_iou = running_iou.double() / len(loader.dataset)

            writers[phase].add_scalar('MSE', epoch_loss, epoch)
            writers[phase].add_scalar('IOU', epoch_iou, epoch)

            print(f'{phase} Loss: {epoch_loss:.4f} IOU: {epoch_iou:.4f}')

            # deep copy the model
            if phase != 'train' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val IOU: {best_iou:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(name='resnet50')
    name = 'test'
    model.to(device)
    loaders = get_loaders(batch_size=128)
    optimizer = optim.Adam(model.parameters(), 1e-4)

    best_model = train_model(model, loaders, optimizer, range(100), device, name)
