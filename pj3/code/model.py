import torch
import torch.nn as nn
import torch.nn.functional as F


'''
basic model for point cloud classification
'''
class cls_3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.batch1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.batch2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.batch3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.batch4 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 40)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batch1(x)
        x = self.relu(self.conv2(x))
        x = self.batch2(x)
        x = self.relu(self.conv3(x))
        x = self.batch3(x)
        x = self.relu(self.conv4(x))
        x = self.batch4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x