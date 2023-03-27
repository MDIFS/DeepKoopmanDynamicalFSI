# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pickle
import configparser
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from AutoEncoder import AE,DataIO,FlowDataset,SlidingSampler

"""Set our seed and other configurations for reproducibility."""
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

""" read config file """
setup = configparser.ConfigParser()
setup.read('input.ini')
epochs = int(setup['DeepLearning']['epochs'])
learning_rate = float(setup['DeepLearning']['learning_rate'])
optthresh = float(setup['DeepLearning']['optthresh'])
target_loss  = float(setup['DeepLearning']['target_loss'])
batch_size = int(setup['DeepLearning']['batchsize'])
sliding = int(setup['DeepLearning']['sliding'])
sliding = 1
"""We set the preference about the CFD"""
dt  = float(setup['CFD']['dt'])
mach= float(setup['CFD']['mach'])
iz  = int(setup['CFD']['iz'])

"""We set the start step, the last step, the intervals"""
nst = int(setup['CFD']['nst'])
nls = int(setup['CFD']['nls'])
nin = int(setup['CFD']['nin'])

""" Dataset """
gpaths = setup['CFD']['gpaths']
fpaths = setup['CFD']['fpaths']
dataio = DataIO(nst,nls,nin,gpaths,fpaths,iz)

grids,ibottom = dataio.readgrid()
js,je,ks,ke,ls,le,ite1,ite2,jd,imove = ibottom

# cropped indices
jcuts = [0,je+1  ,1]
kcuts = [0,ke+1-2,1]
lcuts = [0,le+1-60,1]

flows  = dataio.readflow()

# Set Tensor form
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])

test_dataset = FlowDataset(2,jcuts,kcuts,lcuts,flows,transform)

sampler = SlidingSampler(test_dataset,batch_size,sliding)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    sampler = sampler
)

for _,label in test_loader:
    tmp = label

maxstep = int( torch.max(tmp).item() )

print('Start Testing\n')

""" Load models """ 
model = torch.load("trained_model")
batch = next(iter(test_loader))[0].to(torch.float32).to('cuda')
batch = torch.squeeze(batch)
reconstruction = [batch[0]]
step = 0
with torch.no_grad():
    for batch,_ in test_loader:
        print('step = ', step)
        step = step + 1
        batch = batch.to(torch.float32).to('cuda')
        batch = torch.squeeze(batch)

        out = model(batch)
        ind_half = int(out.size(0)/2)
        X_tilde = out[:ind_half]

        reconstruction.append(X_tilde[0].cpu())

    # """ Calc recreated error """
    # recerrors = []
    # for i,Y_prd in enumerate(reconstruction):

    #     recdata = Y_prd.cpu().numpy()
    #     orgdata = orgdatas[i].cpu().numpy() 

    #     # data shape = (batch * channels * height * width)
    #     error_norm = np.linalg.norm(recdata-orgdata,axis=1,ord=1)
    #     org_norm = np.linalg.norm(orgdata,axis=1,ord=1)

    #     recerror = error_norm/org_norm

    #     recerrors.append(recerror)


    # f = open('recerrors.pickle', 'wb')
    # pickle.dump(recerrors, f)

"""## Visualize Results
Let's try to reconstruct some test images using our trained autoencoder.
"""
print('Post')
with torch.no_grad():
    nstepall = np.arange(nst,nls+nin,nin)

    # write grid
    out_gfiles = [
        './grid_z0003'
    ]
    dataio.writegrid(out_gfiles,grids,jcuts,kcuts,lcuts)

    # write flow
    statedic = []

    for i,rec in enumerate(reconstruction):
        batch = rec.cpu().numpy()

        nstep = nstepall[i]
        fname = 'recflows/u3.0/recflow_z{:0=2}_{:0=8}'.format(iz,nstep)

        q = copy.deepcopy( batch )

        dataio.writeflow(fname,q,jcuts,kcuts,lcuts)
