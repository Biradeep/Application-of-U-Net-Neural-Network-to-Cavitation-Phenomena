################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Compute errors for a test set and visualize. This script can loop over a range of models in
# order to compute an averaged evaluation.
#
################

import os,sys,random,math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TurbDataset
from DfpNet import TurbNetG, weights_init
import utils
from utils import log

suffix = "" # customize loading & output if necessary
prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

expo = 5
dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDirTest="/content/gdrive/MyDrive/data/test/")
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)

targets = torch.FloatTensor(1, 3, 128, 128)
targets = Variable(targets)
targets = targets.cuda()
inputs = torch.FloatTensor(1, 3, 128, 128)
inputs = Variable(inputs)
inputs = inputs.cuda()

targets_dn = torch.FloatTensor(1, 3, 128, 128)
targets_dn = Variable(targets_dn)
targets_dn = targets_dn.cuda()
outputs_dn = torch.FloatTensor(1, 3, 128, 128)
outputs_dn = Variable(outputs_dn)
outputs_dn = outputs_dn.cuda()

netG = TurbNetG(channelExponent=expo)
lf = "./" + prefix + "testout{}.txt".format(suffix)
utils.makeDirs(["results_test"])

# loop over different trained models
avgLoss = 0.

avgLoss_p = 0.
avgLoss_v_x = 0.
avgLoss_v_y = 0.

losses = []

losses_p = []
losses_v_x = []
losses_v_y = []

models = []
#print(torch.load('/content/gdrive/MyDrive/Deep-Flow-Prediction/modelG.pth'))
#for si in range(1):
    #s = chr(96+si)
    #if(si==0):
       # s = "" # check modelG, and modelG + char
    #modelFn = "./" + prefix + "modelG{}{}".format(suffix,s)
    #modelFn = "/content/gdrive/MyDrive/Deep-Flow-Prediction/modelG.pth"
    #if not os.path.isfile(modelFn):
        #continue
modelFn = "/content/gdrive/MyDrive/Deep-Flow-Prediction/modelG.pth"


models.append(modelFn)
log(lf, "Loading " + modelFn )
netG = TurbNetG(channelExponent=expo, dropout = 0.5)
netG.load_state_dict( torch.load(r"/content/gdrive/MyDrive/Deep-Flow-Prediction/modelG.pth") )
log(lf, "Loaded " + modelFn )
netG.cuda()

criterionL1 = nn.L1Loss()
criterionL1.cuda()
L1val_accum = 0.0
L1val_dn_accum = 0.0
lossPer_p_accum = 0
lossPer_v_accum = 0
lossPer_v_x_accum = 0
lossPer_v_y_accum = 0

e_p_accum = 0
e_v_x_accum =0
e_v_y_accum =0

lossPer_accum = 0

netG.eval()

for i, data in enumerate(testLoader, 0):
    inputs_cpu, targets_cpu = data
    targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
    inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
    targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

    outputs = netG(inputs)
    outputs_cpu = outputs.data.cpu().numpy()[0]
    targets_cpu = targets_cpu.cpu().numpy()[0]

    lossL1 = criterionL1(outputs, targets)
    L1val_accum += lossL1.item()

    e_p = np.mean(np.abs(outputs_cpu[0] - targets_cpu[0])/np.sum(np.abs(targets_cpu[0])))
    e_v_x = np.mean(np.abs(outputs_cpu[1] - targets_cpu[1]) / np.sum(np.abs(targets_cpu[1])))
    e_v_y = np.mean(np.abs(outputs_cpu[2] - targets_cpu[2]) / np.sum(np.abs(targets_cpu[2])))

    # precentage loss by ratio of means which is same as the ratio of the sum
    lossPer_p = np.sum(np.abs(outputs_cpu[0] - targets_cpu[0]))/np.sum(np.abs(targets_cpu[0]))
    lossPer_v = ( np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) ) / ( np.sum(np.abs(targets_cpu[1])) + np.sum(np.abs(targets_cpu[2])) )

    lossPer_v_x = np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) / np.sum(np.abs(targets_cpu[1]))
    lossPer_v_y = np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) / np.sum(np.abs(targets_cpu[2]))

    lossPer = np.sum(np.abs(outputs_cpu - targets_cpu))/np.sum(np.abs(targets_cpu))

    e_p_accum += e_p.item()
    e_v_x_accum += e_v_x.item()
    e_v_y_accum += e_v_y.item()

    lossPer_p_accum += lossPer_p.item()
    lossPer_v_accum += lossPer_v.item()

    lossPer_v_x_accum += lossPer_v_x.item()
    lossPer_v_y_accum += lossPer_v_y.item()

    lossPer_accum += lossPer.item()

    #log(lf, "Test sample %d"% i )
    #log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])), lossPer_p.item()) )
    #log(lf, "    velocity:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) , lossPer_v.item() ) )

    #log(lf, "    velocity x:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])), lossPer_v_x.item()))
    #log(lf, "    velocity y:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])), lossPer_v_y.item()))

    #log(lf, "    aggregate: abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu    - targets_cpu   )), lossPer.item()) )

    # Calculate the norm
    input_ndarray = inputs_cpu.cpu().numpy()[0]
    v_norm = ( np.max(np.abs(input_ndarray[0,:,:]))**2 + np.max(np.abs(input_ndarray[1,:,:]))**2 )**0.5

    outputs_denormalized = dataset.denormalize(outputs_cpu, v_norm)
    targets_denormalized = dataset.denormalize(targets_cpu, v_norm)

    # denormalized error
    outputs_denormalized_comp=np.array([outputs_denormalized])
    outputs_denormalized_comp=torch.from_numpy(outputs_denormalized_comp)
    targets_denormalized_comp=np.array([targets_denormalized])
    targets_denormalized_comp=torch.from_numpy(targets_denormalized_comp)

    targets_denormalized_comp, outputs_denormalized_comp = targets_denormalized_comp.float().cuda(), outputs_denormalized_comp.float().cuda()

    outputs_dn.data.resize_as_(outputs_denormalized_comp).copy_(outputs_denormalized_comp)
    targets_dn.data.resize_as_(targets_denormalized_comp).copy_(targets_denormalized_comp)

    lossL1_dn = criterionL1(outputs_dn, targets_dn)
    L1val_dn_accum += lossL1_dn.item()

    # write output image, note - this is currently overwritten for multiple models
    os.chdir("./results_test/")
    utils.imageOut("%04d"%(i), outputs_cpu, targets_cpu, normalize=False, saveMontage=True) # write normalized with error
    os.chdir("../")

log(lf, "\n")

e_p_accum /= len(testLoader)
e_v_x_accum /= len(testLoader)
e_v_y_accum /= len(testLoader)

L1val_accum     /= len(testLoader)
lossPer_p_accum /= len(testLoader)
lossPer_v_accum /= len(testLoader)

lossPer_v_x_accum /= len(testLoader)
lossPer_v_y_accum /= len(testLoader)


lossPer_accum   /= len(testLoader)
L1val_dn_accum  /= len(testLoader)

log(lf, "Loss percentage (p, v, v_x, v_y, combined): %f %%    %f %%    %f %%   %f %%   %f %%" % (lossPer_p_accum*100, lossPer_v_accum*100, lossPer_v_x_accum*100, lossPer_v_y_accum*100, lossPer_accum*100 ) )
log(lf, "L1 error: %f" % (L1val_accum) )
log(lf, "Denormalized error: %f" % (L1val_dn_accum) )
log(lf, "\n")

log(lf, "e_p error: %f" % (e_p_accum) )
log(lf, "e_v_x error: %f" % (e_v_x_accum) )
log(lf, "e_v_y error: %f" % (e_v_y_accum) )
log(lf, "e_avg: %f" % ((e_p_accum + e_v_x_accum + e_v_y_accum)/3))
log(lf, "\n")

avgLoss_p += lossPer_p_accum
avgLoss_v_x += lossPer_v_x_accum
avgLoss_v_y += lossPer_v_y_accum
losses_p.append(lossPer_p_accum)
losses_v_x.append(lossPer_v_x_accum)
losses_v_y.append(lossPer_v_y_accum)

avgLoss += lossPer_accum
losses.append(lossPer_accum)


avgLoss /= len(losses)
avgLoss_p /= len(losses_p)
avgLoss_v_x /= len(losses_v_x)
avgLoss_v_y /= len(losses_v_y)
lossStdErr = np.std(losses) / math.sqrt(len(losses))



log(lf, "Averaged relative error and std dev:   %f , %f " % (avgLoss, lossStdErr))
log(lf, "Averaged relative error P: %f " % (avgLoss_p))
log(lf, "Averaged relative error v_x: %f " % (avgLoss_v_x))
log(lf, "Averaged relative error v_y: %f " % (avgLoss_v_y))