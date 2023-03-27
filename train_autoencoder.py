# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

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
lcuts = [0,le+1-60,1]

flows  = dataio.readflow()

# Set Tensor form
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])

train_dataset = FlowDataset(2,jcuts,kcuts,lcuts,flows,transform)


sampler = SlidingSampler(train_dataset,batch_size,sliding,shuffle=True)

shift,scale = sampler.calc_shift_scale()

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler = sampler,
    num_workers = 2
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
                        # eps=1e-6,\
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
                                                 patience=15,\
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
count_decay = 0.0
losses = []
old_loss = 0.0
best_loss = 1.0e5
for epoch in range(epochs):
    loss = 0

    for batch_features,_ in train_loader:
        batch_features = torch.squeeze(batch_features)
        batch = batch_features.to(torch.float32).to('cuda')

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # input normalization
        for i in range(5):
            batch[:,i,:,:] = (batch[:,i,:,:]-shift[i])/(scale[i]+1.0e-11)

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

    if loss < best_loss:
        """ Save models """ 
        torch.save(model, './trained_model')
        best_loss =  loss
        if loss <= target_loss: break

    # compute the learning erros
    scheduler.step(loss)
    # if (old_loss - loss) < 0.01:
    #     if epoch > 100:
    #         decay_rate = 0.9
    #         count_decay += 1
    #         learning_rate = max(1.0e-9,learning_rate_ini*(decay_rate**count_decay))

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
