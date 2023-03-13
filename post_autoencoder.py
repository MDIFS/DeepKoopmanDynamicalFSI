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
batch_size = int(setup['DeepLearning']['batchsize'])
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

test_loader = torch.utils.data.DataLoader(
    test_dataset,batch_size=batch_size,shuffle=None,drop_last=True
)
print('Start Testing\n')

""" Load models """ 
model = torch.load("train_model")

with torch.no_grad():
    num_batches = 0
    orgdatas = []
    for test,_ in test_loader:
        num_batches += 1
        orgdatas.append(test)

    batch = orgdatas[0]
    batch = batch.to(torch.float32).to('cuda')

    reconstruction = []
    for i in range(num_batches):
        out = model(batch)
        ind_half = int(out.size(0)/2)

        X_prd = out[:(ind_half+1)]

        reconstruction.append(X_prd)
        batch = X_prd

    """ Calc recreated error """
    recerrors = []
    for i,Y_prd in enumerate(reconstruction):

        recdata = Y_prd.cpu().numpy()
        orgdata = orgdatas[i].cpu().numpy() 

        # data shape = (batch * channels * height * width)
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
    nstepall = np.arange(nst+nin,nls,nin)

    # write grid
    out_gfiles = [
        './grid_z0001'
    ]
    dataio.writegrid(out_gfiles,grids,jcuts,kcuts,lcuts)

    # write flow
    statedic = []

    for i in range(num_batches):
        batches = reconstruction[i]
        for step in range(batches.size(0)):
            fname = 'recflows/recflow_z{:0=2}_{:0=8}'.format(iz,i*batches.size(0)+step)
    
            q  = batches[i].cpu().numpy()
            dataio.writeflow(fname,q,jcuts,kcuts,lcuts)
