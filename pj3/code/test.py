# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:34:33 2020

@author: yuansiyu
"""

import torch
import torch.nn as nn

from dataset import ModelNetDataset
from torch.utils.data import DataLoader
import numpy as np
from model import cls_3d

def get_accuracy(preds, y):
    preds_np = preds.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    preds_np = np.argmax(preds_np, axis=1)
    correct = (preds_np == y_np).astype(int).astype(float)
    acc = correct.sum()/len(correct)
    return acc


def evaluate(model, dataloader, loss_fn):
    epoch_loss, epoch_acc = 0., 0.
    model.eval()
    total_len = 0.

    for i, batch in enumerate(dataloader):
        points = batch['points']
        label = batch['label']
        points = points.to(device)
        label = label.long().to(device).squeeze(0).view(-1)
        preds = model(points)
        
        loss = loss_fn(preds, label)
        acc = get_accuracy(preds, label)
        
        epoch_acc += acc.item() * len(label)
        epoch_loss += loss.item() * len(label)
        total_len += len(label)
        
    return epoch_loss / total_len, epoch_acc / total_len

if __name__=='__main__':
    
    root = 'modelnet40_ply_hdf5_2048'

    train_data_list = 'train_files.txt'
    test_data_list = 'test_files.txt'

    test_dataset = ModelNetDataset(root, test_data_list)
    
    batch_size = 2
    
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    
    model = cls_3d()
    model.load_state_dict(torch.load('PJ3_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    
    model = model.to(device)
    
    loss_fn = loss_fn.to(device)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    
    
    print("test loss:",test_loss, "test acc:", test_acc)