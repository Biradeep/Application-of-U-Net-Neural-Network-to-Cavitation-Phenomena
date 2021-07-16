import pretrainedmodels
import torch
import torchvision
import timm
import efficientnet_pytorch
import segmentation_models_pytorch as smp
import tarfile
import glob
from os import listdir
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math


import os, sys, random

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from DfpNet import TurbNetG, weights_init
import dataset
import utils
import matplotlib.pyplot as plt


# number of training iterations
iterations = 1000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 3
# data set config
prop=None # by default, use all from "../data/train"
#prop=[1000,0.75,0,0.25] # mix data from multiple directories
# save txt files with per epoch loss?
saveL1 = False

##########################

prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout    = 0.      # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
doLoad     = ""      # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

seed = random.randint(0, 2**32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# create pytorch data object with dfp dataset
data = dataset.TurbDataset(prop, shuffle=1)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
# setup training
epochs = int(iterations/len(trainLoader) + 0.5)
netG = TurbNetG(channelExponent=expo, dropout=dropout)
#print(netG) # print full net

for x,y in trainLoader:
    break


ENCODER = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'identity'
input_parameters = [10, 3, 128, 128]


model = smp.UnetPlusPlus(encoder_name=ENCODER,encoder_depth=4, encoder_weights=ENCODER_WEIGHTS, decoder_channels=input_parameters, in_channels=3, classes=3, activation=ACTIVATION)
print(model)