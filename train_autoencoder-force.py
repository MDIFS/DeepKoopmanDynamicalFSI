# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import configparser
from distutils.util import strtobool
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.cuda.amp import autocast, GradScaler

from ForceAutoEncoder import FAE,DataIO,ForceDataset,CustomLoss,SlidingSampler

"""Set our seed and other configurations for reproducibility."""
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

"""read config file"""
setup = configparser.ConfigParser()
setup.read('input.ini')
relearning = strtobool(setup['General']['relearning_force'])
epochs = int(setup['DeepLearning_force']['epochs'])
batch_size = int(setup['DeepLearning']['batchsize'])
window_size = int(setup['DeepLearning']['batchsize'])
learning_rate = float(setup['DeepLearning_force']['learning_rate'])
optthresh = float(setup['DeepLearning']['optthresh'])
target_loss  = float(setup['DeepLearning']['target_loss'])
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

""" Dataset"""
gpaths = setup['CFD']['gpaths']
fpaths = setup['CFD']['fpaths']
fmpaths = setup['Control']['fmpaths']

dataio = DataIO(nst,nls,nin,gpaths,fpaths,iz,fmpaths)

grids,ibottom = dataio.readgrid()
js,je,ks,ke,ls,le,ite1,ite2,jd,imove = ibottom

# cropped indices
jcuts = [0,je+1  ,1] 
kcuts = [0,ke+1-2,1]
lcuts = [0,le+1-100,1]

# read forces
forces = dataio.readformom(0) # 0 : Only CL

control_inp = None
if control: control_inp = dataio.readformom(inptype)

# Set Tensor form
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])

train_dataset = ForceDataset(2,jcuts,kcuts,lcuts,forces,window_size,sliding,control_inp,control,transform)

sampler = SlidingSampler(train_dataset,batch_size,sliding,shuffle=True)

#shift,scale = sampler.calc_shift_scale()
#fmin,fmax = sampler.calc_min_max()

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler = sampler#,
#    num_workers = 2
)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = FAE().to(device)
# model = Identity().to(device) # for debbug

# create an optimizer object
optimizer = optim.AdamW(model.parameters(),\
                        lr=learning_rate,\
                        # eps=1e-4,\
                        # amsgrad=True,\
                        weight_decay=1.0e0)
# optimizer = optim.Adam(model.parameters(),\
#                        lr=learning_rate,\
#                        # eps=1e-3,\
#                        # amsgrad=True,\
#                        weight_decay=1.0e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=1.0e1)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,\
                                                 mode='min',\
                                                 factor=0.5, \
                                                 patience=1,\
                                                 threshold=optthresh,\
                                                 threshold_mode='rel',\
                                                 cooldown=0, \
                                                 min_lr=1.e-9,\
                                                 eps=1e-08,\
                                                 verbose=True )

# Frobenius norm loss
criterion = CustomLoss()

if relearning:
    trained_model = torch.load('out_force.pt')
    model.load_state_dict(trained_model['model_state_dict'])
    optimizer.load_state_dict(trained_model['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

""" Define GradScaler """
scaler = GradScaler()  # point1: Scaling the gradient information

"""We train our autoencoder for our specified number of epochs."""
count_decay = 0.0
losses = []
old_loss = 0.0
npass = 0
if relearning:
    estart = trained_model['epoch']+1
    epochs = epochs + estart
    best_loss = trained_model['best_loss']
else:    
    estart = 0
    best_loss = 1.0e5

# torch.autograd.set_detect_anomaly(True)

for epoch in range(estart,epochs):
    loss = 0
    # for batch_features,label,u in train_loader:
    for features in train_loader:
        batch = features[0]
        batch = batch.to(torch.float32).to('cuda')

        if control: u = torch.squeeze(features[2]).to(torch.float32).to('cuda')

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # standalization
        shift,scale = sampler.calc_shift_scale(batch)
        batch = (batch-shift)/(scale+1.0e-11)

        # normalization
        # fmin,fmax = sampler.calc_min_max(batch)
        # batch[:,0,0] = (batch[:,0,0]-fmin)/(fmax-fmin)

        # add noise
        # nfactor = 0.1
        # noise = nfactor*torch.normal(mean=0.0,std=torch.std(batch),size=batch.size()).to('cuda')
        # batch_noised = batch + noise
        # noise check
        # fn = (batch + noise).detach().cpu().numpy()
        # fo = batch.detach().cpu().numpy()
        # plt.figure()
        # plt.plot(range(1, len(fn)+1), fn[:,0,0], marker='x')
        # plt.plot(range(1, len(fo)+1), fo[:,0,0], marker='o')
        # plt.title('force')
        # plt.xlabel('step')
        # plt.savefig('noise_check.pdf')

        # compute reconstructions using autocast
        with autocast(True):   # point 2 : automatic selection for presicion of the model
            if control:
                inp = [batch,u]
            else:
                inp = [batch]

            output = model(inp)

            X_batch = torch.squeeze(batch)[:-1]
            Y_batch = torch.squeeze(batch)[1:]
            target  = torch.cat((X_batch,Y_batch),axis=0)

            # compute training reconstruction loss
            train_loss = criterion(output, target)

            if torch.isnan(train_loss).any():
                npass = npass + 1
            else:
                # compute accumulated gradients
                scaler.scale(train_loss).backward()  # point3: using scaled backward function

                # perform parameter update based on current gradients
                scaler.step(optimizer)  # point4: using scaler.step() altered optimizer.step()
        
                # Update Scaler
                scaler.update()  # point 5: update scaler

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            del train_loss
            torch.cuda.empty_cache()

    # compute the epoch training loss
    loss = loss / (len(sampler.batches_indices)-npass)
    losses.append(loss)

    # display the epoch training loss
    print("epoch : {}/{}, train loss = {:.4f}, npass : {}/{}\n".format(epoch + 1, epochs, loss, npass, len(sampler.batches_indices)))

    if loss < best_loss:
        """ Save models """
        outfile = 'out_force.pt'
        torch.save({'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'best_loss':loss,
                    },outfile)
        best_loss =  loss
        torch.save(model,'./learned_model_force')
        if loss <= target_loss:
            print('Loss reached tharget value')
            break

    # compute the learning erros
    scheduler.step(loss)

    old_loss = loss
    npass = 0

""" Output history """
losses = np.array(losses)
plt.figure()
plt.plot(range(1, len(losses)+1), losses, marker='x')
plt.title('Loss(Frobenius norm)')
plt.xlabel('epoch')
plt.savefig('loss_forces.pdf')

np.save('./losses_forces', losses)
