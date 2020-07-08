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
            loss.backward()
            
            temp = model.classifier[4].weight.grad.clone()
            # print(temp)
            grad.append(temp)
            
            pred = prediction.argmax(dim = 1)

            
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        #f, axes = plt.subplots(1, 2, figsize=(15, 3))

        #learning_curve[epoch] /= batches_n
        #axes[0].plot(learning_curve)

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

def l2_dist(grad):
    r = []
    l = len(grad)
    for i in range(l-1):
        g1 = grad[i].cpu().numpy()
        g2 = grad[i+1].cpu().numpy()
        g_norm = np.linalg.norm(g2-g1)
        r.append(g_norm)
    return r

def VGG_Grad_Pred(VGG_A_grads):
    r = []
    l = len(VGG_A_grads)
    for i in range(l):
        temp = l2_dist(VGG_A_grads[i])
        r.append(temp)
    return r
# Train your model
# feel free to modify
epo = 20
loss_save_path = ''
grad_save_path = ''


set_random_seeds(seed_value=2020, device=device)
print('----First model----'+'\n')

lr_list = [1e-3, 2e-3, 1e-4, 5e-4]
loss_list = []
grad_list = []
for lr in lr_list:
    model = VGG_A()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    VGG_A_loss, VGG_A_grads, val_accuracy_curve = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    #np.savetxt(os.path.join(loss_save_path, 'loss.txt'), VGG_A_loss, fmt='%s', delimiter=' ')
    #np.savetxt(os.path.join(grad_save_path, 'grads.txt'), VGG_A_grads, fmt='%s', delimiter=' ')
    loss_list.append(VGG_A_loss)
    
    grads_l2_dist = VGG_Grad_Pred(VGG_A_grads)
    grad_list.append(grads_l2_dist)


# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []

for epoch in range(epo):
    eplen = len(grad_list[0][epoch])
    for ele in range(eplen):
        max_loss = max(grad_list[0][epoch][ele],grad_list[1][epoch][ele],grad_list[2][epoch][ele],grad_list[3][epoch][ele])
        max_curve.append(max_loss)
        min_loss = min(grad_list[0][epoch][ele],grad_list[1][epoch][ele],grad_list[2][epoch][ele],grad_list[3][epoch][ele])
        min_curve.append(min_loss)
    

print('----Next model----'+'\n')

lr_list = [1e-3, 2e-3, 1e-4, 5e-4]
loss_list = []
grad_list = []
for lr in lr_list:
    model = VGG_A_BatchNorm()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    VGG_A_loss, VGG_A_grads, val_accuracy_curve = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    #np.savetxt(os.path.join(loss_save_path, 'loss.txt'), VGG_A_loss, fmt='%s', delimiter=' ')
    #np.savetxt(os.path.join(grad_save_path, 'grads.txt'), VGG_A_grads, fmt='%s', delimiter=' ')
    loss_list.append(VGG_A_loss)
    
    grads_l2_dist = VGG_Grad_Pred(VGG_A_grads)
    grad_list.append(grads_l2_dist)



# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve_BN = []
max_curve_BN = []
for epoch in range(epo):
    eplen = len(grad_list[0][epoch])
    for ele in range(eplen):
        max_loss = max(grad_list[0][epoch][ele],grad_list[1][epoch][ele],grad_list[2][epoch][ele],grad_list[3][epoch][ele])
        max_curve_BN.append(max_loss)
        min_loss = min(grad_list[0][epoch][ele],grad_list[1][epoch][ele],grad_list[2][epoch][ele],grad_list[3][epoch][ele])
        min_curve_BN.append(min_loss)

def write_file(ls,fname):
    f = open(fname, "w",encoding='UTF-8')
    i = 0
    for ele in ls:
        i = i+1
        f.write(str(ele)+'\t')
        if i % 100 == 0:
            f.write('\n')
    f.close()
    
write_file(min_curve_BN,'min_curve_BN.txt')
write_file(max_curve_BN,'max_curve_BN.txt')
write_file(min_curve,'min_curve.txt')
write_file(max_curve,'max_curve.txt')
# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(min_curve_BN, max_curve_BN, min_curve, max_curve):
    x = list(range(len(min_curve)))
    x = np.array(x) 
    min_curve_BN = np.array(min_curve_BN) 
    max_curve_BN = np.array(max_curve_BN) 
    min_curve = np.array(min_curve) 
    max_curve = np.array(max_curve) 
    
    ax1 = plt.subplot(1, 2, 1, frameon = False)
    plt.plot(x, min_curve, color = '#DB7093')
    plt.plot(x, max_curve, color = '#DB7093')
    p1 = plt.fill_between(x, min_curve, max_curve, facecolor="green", alpha=0.1)
    plt.title('Standard VGG')
    plt.ylabel('grad_landscape')
    plt.xlabel('Steps')
    
    plt.ylim((0, 6))
    
    ax2 = plt.subplot(1, 2, 2, frameon = False)
    plt.plot(x, min_curve_BN, color = '#98FB98')
    plt.plot(x, max_curve_BN, color = '#98FB98')
    p2 = plt.fill_between(x, min_curve_BN, max_curve_BN, facecolor="red", alpha=0.1)
    
    
    plt.title('Standard VGG + BatchNorm')
    plt.ylabel('grad_landscape')
    plt.xlabel('Steps')
    
    plt.ylim((0, 6))
    plt.savefig("grad_landscape.jpg")

plot_loss_landscape(min_curve_BN, max_curve_BN, min_curve, max_curve)