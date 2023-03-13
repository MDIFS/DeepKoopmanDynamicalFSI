# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchsummary import summary
from AutoEncoder import AE,Identity,DataIO,FlowDataset,CustomLoss

"""Set our seed and other configurations for reproducibility."""
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

""" read config file """
setup = configparser.ConfigParser()
setup.read('input.ini')
epochs = int(setup['DeepLearning']['epochs'])
batch_size = int(setup['DeepLearning']['batchsize'])
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

""" Dataset"""
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

train_dataset = FlowDataset(2,jcuts,kcuts,lcuts,flows,transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,shuffle=None,drop_last=True
)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE().to(device)
# model = Identity().to(device) # for debbug

# create an optimizer object
# optimizer = optim.AdamW(model.parameters(),\
#                         lr=learning_rate,\
#                         # eps=1e-6,\
#                         # amsgrad=True,\
#                         weight_decay=1.0e-3)
optimizer = optim.Adam(model.parameters(),\
                       lr=learning_rate,\
                       eps=1e-3,\
                       amsgrad=True,\
                       weight_decay=1.0e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=1.0e1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,\
                                                 mode='min',\
                                                 factor=0.1, \
                                                 patience=50,\
                                                 threshold=optthresh,\
                                                 threshold_mode='rel',\
                                                 cooldown=0, \
                                                 min_lr=1.e-8,\
                                                 eps=1e-08,\
                                                 verbose=True )

# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)

# Frobenius norm loss
criterion = CustomLoss()

"""We train our autoencoder for our specified number of epochs."""
losses = []
best_loss = 0.0
for epoch in range(epochs):
    loss = 0
    for batch_features,_ in train_loader:
        batch = batch_features.to(torch.float32).to('cuda')

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        output = model(batch)

        X_batch = batch[:-1]
        Y_batch = batch[1:]
        target  = torch.cat((X_batch,Y_batch),axis=0)

        # compute training reconstruction loss
        train_loss = criterion(output, target)
 
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)
    losses.append(loss)
    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}\n".format(epoch + 1, epochs, loss))
    if loss < best_loss:
        """ Save models """ 
        torch.save(model, './train_model')

        if loss <= target_loss: break

    # compute the learning erros
    scheduler.step(loss)

""" Output history """
losses = np.array(losses)
plt.figure()
plt.plot(range(1, len(losses)+1), losses, marker='x')
plt.title('Loss(Frobenius norm)')
plt.xlabel('epoch')
plt.savefig('loss.pdf')

np.save('./losses', losses)
