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

from torchsummary import summary
from AutoEncoder import AE,Identity,DataIO,FlowDataset,CustomLoss,SlidingSampler

"""Set our seed and other configurations for reproducibility."""
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

""" read config file """
setup = configparser.ConfigParser()
setup.read('input.ini')

epochs = int(setup['DeepLearning']['epochs'])
batch_size = int(setup['DeepLearning']['batchsize'])
learning_rate = float(setup['DeepLearning']['learning_rate'])
learning_rate_ini = learning_rate 
optthresh = float(setup['DeepLearning']['optthresh'])
target_loss  = float(setup['DeepLearning']['target_loss'])
sliding = int(setup['DeepLearning']['sliding'])

control = strtobool(setup['Control']['control'])
inptype = int(setup['Control']['inptype'])

"""We set the preference about the CFD"""
dt  = float(setup['CFD']['dt'])
mach= float(setup['CFD']['mach'])
iz  = int(setup['CFD']['iz'])

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

# read flow
flows  = dataio.readflow()
control_inp = None
if control: control_inp = dataio.readformom(inptype)

# Set Tensor form
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])

train_dataset = FlowDataset(2,jcuts,kcuts,lcuts,flows,control_inp,control,transform)

sampler = SlidingSampler(train_dataset,batch_size,sliding,shuffle=True)

shift,scale = sampler.calc_shift_scale()

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler = sampler#,
#    num_workers = 2
)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE().to(device)
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

""" Define GradScaler """
scaler = GradScaler()  # point1: Scaling the gradient information

"""We train our autoencoder for our specified number of epochs."""
count_decay = 0.0
losses = []
old_loss = 0.0
best_loss = 1.0e5
torch.autograd.set_detect_anomaly(True)

for epoch in range(epochs):
    loss = 0
    # for batch_features,label,u in train_loader:
    for features in train_loader:
        batch = features[0]
        batch = torch.squeeze(batch)
        batch = batch.to(torch.float32).to('cuda')
        if control: u = torch.squeeze(features[2]).to(torch.float32).to('cuda')

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # standalization 
        for i in range(5):
            batch[:,i,:,:] = (batch[:,i,:,:]-shift[i])/(scale[i]+1.0e-11)

        # compute reconstructions using autocast
        with autocast(False):   # point 2 : automatic selection for presicion of the model

            if control:
                inp = [batch,u]
            else:
                inp = [batch]

            output = model(inp)

            X_batch = batch[:-1]
            Y_batch = batch[1:]
            target  = torch.cat((X_batch,Y_batch),axis=0)

            # compute training reconstruction loss
            train_loss = criterion(output, target)
 
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
    loss = loss / len(train_loader)
    losses.append(loss)

    if loss < best_loss:
        """ Save models """ 
        torch.save(model, './trained_model')
        best_loss =  loss
        if loss <= target_loss:
            print('Loss reached tharget value')
            break

    # compute the learning erros
    scheduler.step(loss)

    old_loss = loss

    # display the epoch training loss
    print("epoch : {}/{}, train loss = {:.4f}\n".format(epoch + 1, epochs, loss))


""" Output history """
losses = np.array(losses)
plt.figure()
plt.plot(range(1, len(losses)+1), losses, marker='x')
plt.title('Loss(Frobenius norm)')
plt.xlabel('epoch')
plt.savefig('loss.pdf')

np.save('./losses', losses)
