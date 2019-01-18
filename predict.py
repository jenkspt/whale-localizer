from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data

from utils import get_model, get_test_transform, unscale_bbox
from dataset import WhalePredictDataset

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(pretrained=False, name='resnet50')
    model.load_state_dict(torch.load('model.pt'))
    model.to(device)
    model.eval()

    images = Path('../whale-id/data/train').glob('*.jpg')
    results_file = Path('../whale-id/data/train_crops.csv')

    ds = WhalePredictDataset(images, get_test_transform())
    loader = data.DataLoader(ds, 64, num_workers=cpu_count())
    with torch.no_grad(): 
        filenames = []
        bbox_lst = []
        for images, transforms, names in loader:
            transforms = transforms.to(device).to(torch.float)
            images = images.to(device)
            pred = torch.sigmoid(model(images))
            bboxes = unscale_bbox(pred, (224., 128.))
            n = len(bboxes)
            # Add ones to points for affine transform
            bboxes = bboxes.reshape(n,2,2)
            bboxes_pad = torch.cat([bboxes, torch.ones((n,2,1), device=device)],-1)
            bboxes_t = torch.transpose(bboxes_pad,2,1)
            
            # Apply inverse transform to get back to original image coordinates
            bboxes = transforms @ bboxes_t

            bboxes = bboxes[:,:2,:].transpose(2,1).reshape(n,4)

            filenames += list(names)
            bbox_lst.append(bboxes.cpu().numpy())

    df = pd.DataFrame(np.concatenate(bbox_lst, 0), 
            index=filenames, columns=['x1','y1','x2','y2'])
    df.index.name = 'Image'
    df.to_csv(results_file)
