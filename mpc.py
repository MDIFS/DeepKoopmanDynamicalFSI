# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pickle
import configparser
import copy
import subprocess
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.cuda.amp import autocast, GradScaler

# from AutoEncoder import AE,DataIO,FlowDataset,SlidingSampler,FSI
from AutoEncoder import AE
from AutoEncoder import DataIO as dio
from AutoEncoder import FlowDataset as fds
from AutoEncoder import SlidingSampler as ss

from ForceAutoEncoder import FAE
from ForceAutoEncoder import DataIO as dio_force
from ForceAutoEncoder import ForceDataset as forcds
from ForceAutoEncoder import SlidingSampler as ss_force

from ConvxOpt import ConvxOpt, FSI

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
epochs = int(setup['DeepLearning']['epochs'])
learning_rate = float(setup['DeepLearning']['learning_rate'])
optthresh = float(setup['DeepLearning']['optthresh'])
target_loss  = float(setup['DeepLearning']['target_loss'])
batch_size = int(setup['DeepLearning']['batchsize'])
window_size = int(setup['DeepLearning']['batchsize'])
sliding = int(setup['DeepLearning']['sliding'])
fc_features = int(setup['DeepLearning']['full_connected'])

control = strtobool(setup['Control']['control'])
inptype = int(setup['Control']['inptype'])

ured = float(setup['MPC']['ured'])
R    = float(setup['MPC']['R'])

"""We set the preference about the CFD"""
dt  = float(setup['CFD']['dt'])
mach= float(setup['CFD']['mach'])
re  = float(setup['CFD']['re'])
iz  = int(setup['CFD']['iz'])

"""We set the start step, the last step, the intervals"""
nst = int(setup['MPC']['nst'])
nls = int(setup['MPC']['nls'])
nin = int(setup['CFD']['nin'])

""" Dataset """
gpaths = setup['CFD']['gpaths']
fpaths = setup['CFD']['fpaths']
fmpaths= setup['Control']['fmpaths']

""" Set Dynamics """
print('Set Dynamics...\n')
dataio = dio(nst,nls,nin,gpaths,fpaths,iz,fmpaths)

grids,ibottom = dataio.readgrid()
js,je,ks,ke,ls,le,ite1,ite2,jd,imove = ibottom

# cropped indices
jcuts = [0,je+1  ,1]
kcuts = [0,ke+1-2,1]
lcuts = [0,le+1-100,1]

# output cropped grid
dataio.tweak_writegrid(['grid_z0003'],grids,jcuts,kcuts,lcuts)

flows  = dataio.readflow()
control_inp = None
if control: control_inp = dataio.readformom(inptype)
# Set Tensor form
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])

test_dataset = fds(2,jcuts,kcuts,lcuts,flows,control_inp,control,transform)

sampler = ss(test_dataset,batch_size,sliding)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    sampler = sampler
)

orgdatas = []

for batch,label,u in test_loader:
    test = batch[0][0]
    tmp = label
    orgdatas.append(test)


maxstep = int( torch.max(tmp).item() )

print('Set Forces...')
dioforce = dio_force(nst,nls,nin,gpaths,fpaths,iz,fmpaths)
forces = dioforce.readformom(0) # 0 : Only CL
transform_force = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
     ])
test_dataset_force = forcds(2,jcuts,kcuts,lcuts,forces,window_size,sliding,control_inp,control,transform_force)

sampler_force = ss_force(test_dataset_force,window_size,sliding)

test_loader_force = torch.utils.data.DataLoader(
    test_dataset_force,
    sampler = sampler_force
)

print('Start MPC')
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Load models """ 
model = torch.load("learned_model")
model_force = torch.load("learned_model_force")

reconstruction = []
step = nst

""" set instances """
convxopt = ConvxOpt(batch_size,inptype)
horizon  = convxopt.T # horizontal window

fsi = FSI(jcuts,kcuts,lcuts,iz,dataio,mach,re,dt,inptype,ured,horizon)

with torch.no_grad():
    # Initial state variables D_0 (X_0, Y_0)
    features = next( iter(test_loader) ) # D_0

    for icount in range(maxstep):
        print('step = ', step)
        step = step + nin*sliding

        # # Set Fluid Force
        # batch = features[0]
        # batch = torch.squeeze(batch)
        # batch = batch.to(torch.float32).to('cuda')
        if control: u = torch.squeeze(features[2]).to(torch.float32).to('cuda')

        # ## standalized input batches
        # shift = torch.mean(batch,(0,2,3)).to(torch.float32)
        # scale = torch.std(batch,(0,2,3)).to(torch.float32)
        # for i in range(5):
        #     batch[:,i,:,:] = (batch[:,i,:,:] - shift[i])/(scale[i]+1.0e-11)

        # ## compute reconstructions using autocast
        # with autocast(False): # point 2 :automatic selection for precision of the model
        #     if control:
        #         inp = [batch,u]
        #     else:
        #         print('MPC needs control')
        #         exit()

        #     ### Extract gx in latent space and A, B matrices
        #     gx,A,B = model.encoder_forMPC(inp)
        #     cvec = gx[:,:horizon]
        # ## prepare the objective function
        
        
        # exit()
        # ## unstandalized
        # for i in range(5):
        #     X_tilde[:,i,:,:] = X_tilde[:,i,:,:] * (scale[i]+1.0e-11) + shift[i]

        # Deep FSI 
        # forces = fsi.calc_force(X_tilde[:ind_half],u[:ind_half])
        ''' test '''
        fluid_forces = next(iter(test_loader_force))[0].to(torch.float32).to('cuda')
        struct_forces = fsi.structure_force(u,inptype,ured,mach)
        struct_forces = torch.from_numpy(struct_forces)[None].to(torch.float32).to('cuda')
        ''''''''''''
        
        ## map forces into the latent space
        ### map fluid forces
        batch = fluid_forces

        with autocast(False): # point 2 :automatic selection for precision of the model
            if control:
                inp = [batch,u[0]]
            else:
                print('MPC needs control')
                exit()

            ### Extract gx in latent space and A, B matrices
            gx,Af,Bf = model_force.encoder_forMPC(inp)
            cvec_fluid = gx[:,:horizon]

        ### map structure forces
        batch = struct_forces

        with autocast(False): # point 2 :automatic selection for precision of the model
            if control:
                inp = [batch,u[0]]
            else:
                print('MPC needs control')
                exit()

            ### Extract gx in latent space and A, B matrices
            gx,_,_ = model_force.encoder_forMPC(inp)
            cvec_struct = gx[:,:horizon]

        # MPC
        cforces = [fluid_forces,struct_forces]
        u_optim = convxopt.solve_cvx(cforces,R,Af,Bf)
        exit()
        reconstruction.append(X_tilde[0].cpu())

    # """ Calc recreated error """
    recerrors = []
    for i,X_tilde in enumerate(reconstruction):
        recdata = X_tilde.cpu().numpy()
        orgdata = orgdatas[i].cpu().numpy() 

        # data shape = (batch * channels * height * width)
        # error_norm = np.linalg.norm(recdata-orgdata,axis=1,ord=1)
        # org_norm = np.linalg.norm(orgdata,axis=1,ord=1)

        error_norm = np.linalg.norm(recdata-orgdata,axis=0,ord=1)
        org_norm = np.linalg.norm(orgdata,axis=0,ord=1)

        recerror = error_norm/(org_norm)
        recerrors.append(recerror)


    f = open('recerrors.pickle', 'wb')
    pickle.dump(recerrors, f)

"""## Visualize Results
Let's try to reconstruct some test images using our trained autoencoder.
"""
print('Post')
with torch.no_grad():
    nstepall = np.arange(nst,nls+nin,nin*sliding)

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
