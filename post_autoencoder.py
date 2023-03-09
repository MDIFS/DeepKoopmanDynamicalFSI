# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pickle
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from AutoEncoder import AE,DataIO,FlowDataset

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
lcuts = [0,le+1  ,1]

flows  = dataio.readflow()

# Set Tensor form
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])

test_dataset = FlowDataset(2,jcuts,kcuts,lcuts,flows,transform)

print('Start Testing\n')

""" Load models """ 
model = torch.load("train_model")

with torch.no_grad():
    batches_list = [batch[0].to(torch.float32).to('cuda') for batch in test_dataset]

    # add new axis
    for i,batch in enumerate(batches_list):
        batches_list[i] = batch[None]
        
    reconstruction = model(batches_list)

    # Calc recreated error
    recerrors = []
    for i,batch in enumerate(reconstruction):
        recdata = reconstruction[i].cpu().numpy()
        orgdata = batches_list[i].cpu().numpy() 

        # data shape = (Batch * channels * height * width)
        error_norm = np.linalg.norm(recdata-orgdata,axis=1,ord=1)
        org_norm = np.linalg.norm(orgdata,axis=1,ord=1)
        
        recerror = error_norm/org_norm
        recerrors.append(recerror)

    f = open('recerrors.pickle', 'wb')
    pickle.dump(recerrors, f)

"""## Visualize Results
Let's try to reconstruct some test images using our trained autoencoder.
"""
print('Post')
with torch.no_grad():
    nstepall = np.arange(nst,nls,nin)

    # write grid
    out_gfiles = [
        './grid_z0001'
    ]
    dataio.writegrid(out_gfiles,grids,jcuts,kcuts,lcuts)

    # write flow
    statedic = []
    
    for i, step in enumerate(nstepall[:-1]):
        fname = 'recflows/recflow_z{:0=2}_{:0=8}'.format(iz,step)
        q  = reconstruction[i][0].cpu().numpy()

        dataio.writeflow(fname,q,jcuts,kcuts,lcuts)
