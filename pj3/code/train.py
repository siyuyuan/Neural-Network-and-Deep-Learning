# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:19:05 2020

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


def train(model, dataloader, optimizer, loss_fn):
    epoch_loss, epoch_acc = 0.,0.
    model.train()
    total_len = 0
    for i, batch in enumerate(dataloader):
        points = batch['points']
        label = batch['label']
        points = points.to(device)
        label = label.long().to(device).squeeze(0).view(-1)
        preds = model(points)
        #label = label
        
        #print(preds)
        #print(label)
        
        loss = loss_fn(preds, label)
        acc = get_accuracy(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_acc += acc.item() * len(label)
        epoch_loss += loss.item() * len(label)
        total_len += len(label)
        
    return epoch_loss / total_len, epoch_acc / total_len


if __name__=='__main__':
    
    root = 'modelnet40_ply_hdf5_2048'

    train_data_list = 'train_files.txt'
    test_data_list = 'test_files.txt'
    train_dataset = ModelNetDataset(root, train_data_list)
    test_dataset = ModelNetDataset(root, test_data_list)
    
    batch_size = 32
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    
    model = cls_3d()
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    num_epochs = 100
    best_acc = 0.
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader,optimizer, loss_fn)
        
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), 'PJ3_model.pth')
        
        print("epoch",epoch,"train loss:",train_loss, "train acc:", train_acc)

