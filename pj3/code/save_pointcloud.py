# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:31:49 2020

@author: yuansiyu
"""
from dataset import ModelNetDataset
from torch.utils.data import Dataset, DataLoader
import torch
#save_pointcloud.py

if __name__ == '__main__':
    root = 'D:\\modelnet40_ply_hdf5_2048'

    train_data_list = 'train_files.txt'
    test_data_list = 'test_files.txt'
    train_dataset = ModelNetDataset(root, train_data_list)
    test_dataset = ModelNetDataset(root, test_data_list)
    
    batch_size = 1
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    
    it = iter(train_loader)
    batch = next(it)
    print(batch)
    points= batch['points']
    
    points = points.squeeze(0).permute(1,0)
    points_ls = points.numpy()
    f = open('pointcloud.obj', "w",encoding='UTF-8')
    for point in points_ls:
        f.write('v'+ ' ' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]))
        f.write('\n')
    f.close()
    