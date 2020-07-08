import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#print(torch.cuda.get_device_name(3))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

for X,y in train_loader:
    print(X[0])
    print(y[0])
    print(X[0].shape)
    img = np.transpose(X[0], [1,2,0])
    plt.imshow(img*0.5 + 0.5)
    plt.savefig('sample.png')
    print(X[0].max())
    print(X[0].min())
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(pred,y):
    return pred.eq(y.view_as(pred)).sum().item()

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [0] * epochs_n
    train_accuracy_curve = [0] * epochs_n
    val_accuracy_curve = [0] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            loss_list.append(loss.item())
            temp = model.classifier[4].weight.grad
            grad.append(temp)
            pred = prediction.argmax(dim = 1)

            loss.backward()
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        model.eval()
        batches_n = len(val_loader.dataset)
        for data in val_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            pred = prediction.argmax(dim = 1)
            val_accuracy_curve[epoch] += get_accuracy(pred,y)
            
        val_accuracy_curve[epoch]  = val_accuracy_curve[epoch] /batches_n
        if max_val_accuracy < val_accuracy_curve[epoch]:
            max_val_accuracy = val_accuracy_curve[epoch]
            max_val_accuracy_epoch = epoch
        
        print("epoch:{}, valid accuracy:{}, max valid accuracy:{}, max valid accuracy epoch:{}".format(epoch, val_accuracy_curve[epoch], max_val_accuracy,max_val_accuracy_epoch))
    

    return losses_list, grads, val_accuracy_curve


# Train your model
# feel free to modify
epo = 20
loss_save_path = ''
grad_save_path = ''


set_random_seeds(seed_value=2020, device=device)

print('----First model for picture----'+'\n')
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
VGG_A_loss, VGG_A_grads, val_accuracy_curve = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
x = range(0, epo)
y = val_accuracy_curve
plt.plot(x, y, 'o-')
plt.title('valid accuracy vs epoches')
plt.ylabel('valid accuracy')
plt.xlabel('epoches')
plt.savefig("VGG_A_accuracy.jpg")

print('----next model for picture----'+'\n')
model = VGG_A_BatchNorm()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
VGG_A_loss, VGG_A_grads, val_accuracy_curve = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
x = range(0, epo)
y = val_accuracy_curve
plt.plot(x, y, 'o-')
plt.title('valid accuracy vs epoches')
plt.ylabel('valid accuracy')
plt.xlabel('epoches')
plt.savefig("VGG_A_BatchNorm_accuracy.jpg")

