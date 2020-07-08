import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as tud
import time
import matplotlib.pyplot as plt

class Res_Net(nn.Module):

    def __init__(self):
        super(Res_Net,self).__init__()
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        
       
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        self.res_connect2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        self.res_connect3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        self.res_connect4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=10, bias=True)

        

    def forward(self,x):
        x = self.layer0(x)
        
        x = self.layer1(x) + x
        x = self.layer2(x) + self.res_connect2(x)
        x = self.layer3(x) + self.res_connect3(x)
        x = self.layer4(x) + self.res_connect4(x)
        
        x = self.avgpool(x)
        x = x.view(-1,512)
        x = self.fc(x)

        return F.log_softmax(x,dim=1)

def train(model, device, train_dataloader, optimizer, epoch, loss_fn):
    model.train()

    for idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        preds = model(data) # batch_size * 10
        #loss = loss_fn(preds, target)
        loss = F.nll_loss(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1000 == 0:
            print("Train Epoch:{}, iteration:{}, Loss:{}".format(epoch, idx, loss.item()))

def evaluate(model, device, valid_dataloader,loss_fn, flag):
    model.eval()
    total_loss =0.
    correct = 0.
    total_len = len(valid_dataloader.dataset)
    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_dataloader):
            
            data, target = data.to(device), target.to(device)
            output = model(data) # batch_size * 1
            #total_loss += loss_fn(output, target).item()
            total_loss += F.nll_loss(output, target, reduction = "sum").item()
            pred = output.argmax(dim = 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    total_loss = total_loss / total_len
    acc = correct/total_len
    if flag == 1:
        print("test loss:{}, Accuracy:{}".format(total_loss, acc)) 
    else:
        print("valid loss:{}, Accuracy:{}".format(total_loss, acc)) 
    return total_loss, acc


def evaluate_test(model, device, test_dataloader,loss_fn, flag):
    total_loss =0.
    correct = 0.
    total_len = len(test_dataloader.dataset)
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            
            data, target = data.to(device), target.to(device)
            output = model(data) # batch_size * 1
            #total_loss += loss_fn(output, target).item()
            total_loss += F.nll_loss(output, target, reduction = "sum").item()
            pred = output.argmax(dim = 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    total_loss = total_loss / total_len
    acc = correct/total_len
    if flag == 1:
        print("test loss:{}, Accuracy:{}".format(total_loss, acc)) 
    else:
        print("valid loss:{}, Accuracy:{}".format(total_loss, acc)) 

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

validset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
valid_dataloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.03
momentum = 0.5
num_features = 3
hidden_size = 100
output_size = 10
loss_fn = nn.CrossEntropyLoss()

model = Res_Net().to(device)
model.apply(init_weights) 

optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay=0.001)

starttime = time.time()
num_epochs = 100

total_loss = []
acc = []

train(model, device, train_dataloader, optimizer, 0, loss_fn)
total_loss_0, acc_0 = evaluate(model, device, valid_dataloader,loss_fn, 0)
torch.save(model.state_dict(),"CIFAR10_cnn.pth")    
total_loss.append(total_loss_0)
acc.append(acc_0)

for epoch in range(1,num_epochs):
    train(model, device, train_dataloader, optimizer, epoch, loss_fn)
    total_loss_0, acc_0 = evaluate(model, device, valid_dataloader,loss_fn, 0)
    if total_loss_0 < min(total_loss) and acc_0 > max(acc):
        torch.save(model.state_dict(),"CIFAR10_cnn.pth")
    total_loss.append(total_loss_0)
    acc.append(acc_0)

model_ready = Res_Net().to(device)
model_ready.load_state_dict(torch.load('CIFAR10_cnn.pth'))
evaluate_test(model_ready, device, test_dataloader,loss_fn, 1)

endtime = time.time()
dtime = endtime - starttime
print("run time:%.8s s" % dtime)

x1 = range(0, num_epochs)
x2 = range(0, num_epochs)
y1 = acc
y2 = total_loss
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('valid accuracy vs. epoches')
plt.ylabel('valid accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('valid loss vs. epoches')
plt.ylabel('valid loss')
plt.savefig("Res_Net_accuracy_loss.jpg")