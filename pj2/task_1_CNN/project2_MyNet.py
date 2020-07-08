import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as tud
import time
import matplotlib.pyplot as plt

class MyNet(nn.Module):
    def __init__(self,num_features,hidden_size,output_size):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(num_features, 20, 5, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(5*5*50, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # x:1 * 28 * 28
        
        x = self.bn1(self.conv1(x))
        x = F.ReLU(x) # 20 * 28 * 28
        x = F.max_pool2d(x, 2, 2) # 20 * 14 * 14
        
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = F.ReLU(x) # 20 * 28 * 28
        x = F.max_pool2d(x, 2, 2) # 50 * 5 * 5
        
        x = x.view(-1, 5*5*50) #reshape
        x1 = F.ReLU(self.fc1(x))
        
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)
        
        return F.log_softmax(x1, dim = 1) # log probability

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

model = MyNet(num_features,hidden_size,output_size).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay=0.001)

starttime = time.time()
num_epochs = 50
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

model_ready = MyNet(num_features,hidden_size,output_size).to(device)
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
plt.savefig("MyNet_accuracy_loss.jpg")