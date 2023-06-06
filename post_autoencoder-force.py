# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pickle
import configparser
import copy
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.cuda.amp import autocast, GradScaler

from ForceAutoEncoder import FAE,DataIO,ForceDataset,SlidingSampler

"""Set our seed and other configurations for reproducibility."""
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

""" Define GradScaler """
scaler = GradScaler()  # point1: Scaling the gradient information

""" read config file """
setup = configparser.ConfigParser()
setup.read('input.ini')
epochs = int(setup['DeepLearning_force']['epochs'])
learning_rate = float(setup['DeepLearning_force']['learning_rate'])
optthresh = float(setup['DeepLearning']['optthresh'])
target_loss  = float(setup['DeepLearning']['target_loss'])
batch_size = int(setup['DeepLearning']['batchsize'])
window_size = int(setup['DeepLearning']['batchsize'])
sliding = int(setup['DeepLearning']['sliding'])

control = strtobool(setup['Control']['control'])
inptype = int(setup['Control']['inptype'])

"""We set the preference about the CFD"""
dt  = float(setup['CFD']['dt'])
mach= float(setup['CFD']['mach'])
iz  = int(setup['CFD']['iz'])
re  = float(setup['CFD']['re'])

"""We set the start step, the last step, the intervals"""
nst = int(setup['CFD']['nst'])
nls = int(setup['CFD']['nls'])
nin = int(setup['CFD']['nin'])

""" Dataset """
gpaths = setup['CFD']['gpaths']
fpaths = setup['CFD']['fpaths']
fmpaths= setup['Control']['fmpaths']

dataio = DataIO(nst,nls,nin,gpaths,fpaths,iz,fmpaths)

grids,ibottom = dataio.readgrid()
js,je,ks,ke,ls,le,ite1,ite2,jd,imove = ibottom

# cropped indices
jcuts = [0,je+1  ,1]
kcuts = [0,ke+1-2,1]
lcuts = [0,le+1-100,1]

forces  = dataio.readformom(0)

control_inp = None
if control: control_inp = dataio.readformom(inptype)

# Set Tensor form
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])

test_dataset = ForceDataset(2,jcuts,kcuts,lcuts,forces,window_size,sliding,control_inp,control,transform)

sampler = SlidingSampler(test_dataset,batch_size,sliding)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    sampler = sampler
)

orgdatas = []

for batch,label,u in test_loader:
    test = torch.squeeze(batch)[:-1]
    tmp = label
    orgdatas.append(test)

maxstep = int( torch.max(tmp).item() )

print('Start Testing\n')

""" Load models """ 
model = torch.load("learned_model_force")
batch = next(iter(test_loader))[0].to(torch.float32).to('cuda')
batch = torch.squeeze(batch)

reconstruction = []
step = nst
with torch.no_grad():
    # for batch,_ in test_loader:
    for features in test_loader:
        batch = features[0]
        batch = batch.to(torch.float32).to('cuda')

        if control: u = torch.squeeze(features[2]).to(torch.float32).to('cuda')

        print('step = ', step)
        step = step + nin*sliding

        # standalized input batches
        shift,scale = sampler.calc_shift_scale(batch)
        batch = (batch - shift)/(scale+1.0e-11)

        # compute reconstructions using autocast
        with autocast(False): # point 2 :automatic selection for precision of the model
            if control:
                inp = [batch,u]
            else:
                inp = [batch]

            out = model(inp)

            ind_half = int(out.size(0)/2)
            X_tilde = out[:ind_half]

        # unstandalized
        X_tilde[:] = X_tilde[:] * (scale+1.0e-11) + shift

        reconstruction.append(X_tilde[:])

    # """ Calc recreated error """
    recerrors = []
    for i,X_tilde in enumerate(reconstruction):
        recdata = X_tilde.cpu().numpy()
        orgdata = orgdatas[i].cpu().numpy() 

        # data shape = (batch * channels * height * width)
        # error_norm = np.linalg.norm(recdata-orgdata,axis=1,ord=1)
        # org_norm = np.linalg.norm(orgdata,axis=1,ord=1)

        error_norm = np.linalg.norm(recdata-orgdata,ord=1)
        org_norm = np.linalg.norm(orgdata,ord=1)

        recerror = error_norm/(org_norm)
        print(recerror)
        recerrors.append(recerror)


    f = open('recerrors_forces.pickle', 'wb')
    pickle.dump(recerrors, f)
