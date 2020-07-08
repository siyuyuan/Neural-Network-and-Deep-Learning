# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:24:55 2020

@author: yuansiyu
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_landscape(min_curve_BN, max_curve_BN, min_curve, max_curve):
    x = list(range(len(min_curve)))
    x = np.array(x) 
    min_curve_BN = np.array(min_curve_BN) 
    max_curve_BN = np.array(max_curve_BN) 
    min_curve = np.array(min_curve) 
    max_curve = np.array(max_curve) 
    
    plt.plot(x, min_curve, 'g')
    plt.plot(x, max_curve, 'g')
    p1 = plt.fill_between(x, min_curve, max_curve, facecolor="green", alpha=0.3)
    
    plt.plot(x, min_curve_BN, 'r')
    plt.plot(x, max_curve_BN, 'r')
    p2 = plt.fill_between(x, min_curve_BN, max_curve_BN, facecolor="red", alpha=0.3)
    
    l1 = plt.legend([p1, p2], ["Standard VGG", "Standard VGG + BatchNorm"], loc='upper right')
    plt.title('Loss_landscape vs Steps')
    plt.ylabel('Loss_landscape')
    plt.xlabel('Steps')
    plt.savefig("Loss_landscape_update.jpg")
    plt.gca().add_artist(l1)

def ReadTxtName(address):
    f = open(address, encoding='UTF-8')
    line = f.readline()
    ls = []
    while line:
        line_ = line.replace('\n','')
        line_ = line_.split('\t')
        line_ = line_[:-1]
        line_ = list(map(float,line_))
        ls = ls + line_
        line = f.readline()
    f.close()
    return ls

min_curve_BN = ReadTxtName('min_curve_BN.txt') 
max_curve_BN = ReadTxtName('max_curve_BN.txt')
min_curve = ReadTxtName('min_curve.txt')
max_curve = ReadTxtName('max_curve.txt')
plot_loss_landscape(min_curve_BN, max_curve_BN, min_curve, max_curve)