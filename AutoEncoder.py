# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.sampler
import torchvision
import glob
from readwritePLOT3D import checkheader, readgrid, writegrid, \
                                         readflow, writeflow
 
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        setup = configparser.ConfigParser()
        setup.read('input.ini')
        fc_features = int(setup['DeepLearning']['full_connected'])
        
        #=== Encoder  ===
        enresidualblock = self.EncoderResidualBlock
        self.eblock1 = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.ReLU(True),
            nn.Conv2d(5,256,kernel_size=3,stride=2,padding=1,bias=True)
        )
        self.eblock2 = nn.Sequential(
            enresidualblock(256,256)
        )
        self.eblock23 = nn.Sequential(
            nn.Conv2d(256,128,stride=2,kernel_size=1,padding=1,bias=True)
        )
        self.eblock3 = nn.Sequential(
            enresidualblock(128,128)
        )        
        self.eblock34 = nn.Sequential(
            nn.Conv2d(128,64,stride=2,kernel_size=3,padding=1,bias=True)
        )
        self.eblock4 = nn.Sequential(
            enresidualblock(64,64)
        )
        self.eblock45 = nn.Sequential(
            nn.Conv2d(64,32,stride=2,kernel_size=3,padding=1,bias=True)
        )
        self.eblock5 = nn.Sequential(
            enresidualblock(32,32)
        )
        self.eblock56 = nn.Sequential(
            nn.Conv2d(32,16,stride=2,kernel_size=3,padding=1,bias=True)
        )
        self.eblock6 = nn.Sequential(
            enresidualblock(16,16)
        )
        self.efc = nn.Sequential(
            nn.Linear(560,fc_features)
        )
        #=== Encoder Block ===

        #=== Decoder Block ===
        deresidualblock = self.DecoderResidualBlock
        self.dfc = nn.Sequential(
            nn.Linear(fc_features,560)
        )
        self.dblock6 = nn.Sequential(
            enresidualblock(16,16)
        )
        self.dblock65 = nn.Sequential(
            nn.ConvTranspose2d(16,32,stride=2,kernel_size=3,padding=1,output_padding=(0,0),bias=True)
        )
        self.dblock5 = nn.Sequential(
            enresidualblock(32,32)
        )
        self.dblock54 = nn.Sequential(
            nn.ConvTranspose2d(32,64,stride=2,kernel_size=3,padding=1,output_padding=(1,1),bias=True)
        )
        self.dblock4 = nn.Sequential(
            enresidualblock(64,64)
        )
        self.dblock43 = nn.Sequential(
            nn.ConvTranspose2d(64,128,stride=2,kernel_size=3,padding=1,output_padding=(0,1),bias=True)
        )
        self.dblock3 = nn.Sequential(
            enresidualblock(128,128)
        )        
        self.dblock32= nn.Sequential(
            nn.ConvTranspose2d(128,256,stride=2,kernel_size=3,padding=1,output_padding=(0,0),bias=True)
        )
        self.dblock2 = nn.Sequential(
            enresidualblock(256,256)
        )
        self.dblock1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,5,stride=2,kernel_size=3,padding=1,output_padding=(0,0),bias=True)
        )
        #=== Decoder Block ===

    class EncoderResidualBlock(nn.Module):
        def __init__(self,in_features,out_features,stride=1,kernel_size=3,padding=1,bias=False):
            super().__init__()
            half_out_features = int(out_features/2)
            self.cnn1 = nn.Sequential(
                nn.BatchNorm2d(in_features),
                nn.ReLU(True),
                nn.Conv2d(in_features,half_out_features,1,stride,0,bias=True)
            )
            self.cnn2 = nn.Sequential(
                nn.BatchNorm2d(half_out_features),
                nn.ReLU(True),
                nn.Conv2d(half_out_features,half_out_features,kernel_size,stride,padding,bias=True)
            )
            self.cnn3 = nn.Sequential(
                nn.BatchNorm2d(half_out_features),
                nn.ReLU(True),
                nn.Conv2d(half_out_features,out_features,1,stride,0,bias=True)
            )

            self.shortcut = nn.Sequential(
            )

        def forward(self,x):
            residual = x
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.cnn3(x)
            x += self.shortcut(residual)

            return x

    class DecoderResidualBlock(nn.Module):
        def __init__(self,in_features,out_features,stride=1,kernel_size=3,padding=1,bias=False):
            super().__init__()
            half_out_features = int(out_features/2)
            self.cnn1 = nn.Sequential(
                nn.BatchNorm2d(in_features),
                nn.ReLU(True),
                nn.Conv2d(in_features,half_out_features,1,stride,0,bias=True)
            )
            self.cnn2 = nn.Sequential(
                nn.BatchNorm2d(half_out_features),
                nn.ReLU(True),
                nn.Conv2d(half_out_features,half_out_features,kernel_size,stride,padding,bias=True)
            )
            self.cnn3 = nn.Sequential(
                nn.BatchNorm2d(half_out_features),
                nn.ReLU(True),
                nn.Conv2d(half_out_features,out_features,1,stride,0,bias=True)
            )

            self.shortcut = nn.Sequential(
            )

        def forward(self,x):
            residual = x
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.cnn3(x)
            x += self.shortcut(residual)

            return x

    def encoderforward(self,x):
        # Block 1
        # print('eblock1 ',x.size())
        x = self.eblock1(x)
        # Block 2 / 256*256
        # print('eblock2 ',x.size())
        x = self.eblock2(x)
        x = self.eblock23(x)
        # Block 3 / 128*128
        # print('eblock3 ',x.size())
        x = self.eblock3(x)
        x = self.eblock34(x)
        # Block 4 / 64*64
        # print('eblock4 ',x.size())
        x = self.eblock4(x)
        x = self.eblock45(x)
        # Block 5
        # print('eblock5 ',x.size())
        x = self.eblock5(x)
        x = self.eblock56(x)
        # Block 6
        # print('eblock6 ',x.size())
        x = self.eblock6(x)
        # Fully Connected
        self.eshape = x.size()
        x = x.view(self.eshape[0],-1)
        # print('reshape ',x.size())
        x = self.efc(x)
        # print('Fully Connected',x.size())

        return x

    def decoderforward(self,x):
        # Fully Connected
        x = self.dfc(x)
        # print('Fully Conneted',x.size())
        x = x.view(-1,self.eshape[1],self.eshape[2],self.eshape[3])
        # Block 6
        # print('dblock6 ',x.size())
        x = self.dblock6(x)
        x = self.dblock65(x)
        # Block 5
        # print('dblock5 ',x.size())
        x = self.dblock5(x)
        # Block 4
        x = self.dblock54(x)
        # print('dblock4 ',x.size())
        x = self.dblock4(x)
        # Block 3
        x = self.dblock43(x)
        # print('dblock3 ',x.size())
        x = self.dblock3(x)
        # Block 2
        x = self.dblock32(x)
        # print('dblock2 ',x.size())
        x = self.dblock2(x)
        # Block 1
        # print('dblock1 ',x.size())
        x = self.dblock1(x)
        # Fin
        # print('Final   ',x.size())
        # exit()
        return x

    def calc_Ytilde_prd(self, inp):

        Xtilde = inp[:-1,:]
        Ytilde = inp[1:,:]

        half_ind = int(inp.size(0)/2)

        # least-square
        m,n = Xtilde[:half_ind,:].size(0),Xtilde.size(1)

        l2_lambda = 1.e1
        Xtil = Xtilde[:half_ind,:]
        Ytil = Ytilde[:half_ind,:]
        # print('Xtil = ',Xtil.size())
        # if m < n:
        #     A = torch.t(Xtil)@torch.linalg.pinv(( Xtil @ torch.t(Xtil) + l2_lambda )) @ Ytil
        # else:
        #     A = torch.linalg.pinv( torch.t(Xtil) @ Xtil + l2_lambda ) @ torch.t(Xtil) @ Ytil
        A = torch.linalg.pinv(Xtil) @ Ytil 
        # print('A = ',A.size())
        # koopman
        A_i = A
        Ytilde_pred = torch.zeros( Ytilde.size() ).to('cuda')
        Xtilde_0 = Xtilde[0,:]
        Xtilde_0 = torch.unsqueeze(Xtilde_0, 0)
        # print('Xtilde_0 = ', Xtilde_0.size())

        for i in range(Ytilde.size(0)):
            Ytilde_pred_i =  Xtilde_0 @ A_i 
            # print('Ytilde_pred = ',Ytilde_pred_i.size())
            Ytilde_pred[i,:] = Ytilde_pred_i

            A_i = A @ A
        # print(Ytilde_pred.size())
        # exit()
        # out = torch.cat((Xtilde, Ytilde_pred), axis =0)
        out = torch.cat((Xtilde, Ytilde), axis =0)

        return out

    def forward(self, inp):

        # Encoder
        # print('Encoder')
        enout = self.encoderforward(inp)

        # least square
        # print('Least square')
        enout = self.calc_Ytilde_prd(enout)
        
        # Decoder
        # print('Decoder')
        deout = self.decoderforward(enout)

        reconstructed = deout

        return reconstructed


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Identity()
        )
        self.decoder = nn.Sequential(
            nn.Identity()
        )
    def forward(self, features):
        # print('Process... ')
        # print('Input shape = ',features.size())
        encoded       = self.encoder(features)
        # print('Encoded shape = ',features.size())
        decoded       = self.decoder(encoded)
        # print('Decoded shape = ',features.size())
        reconstructed = decoded
        # print('Finalze Process... \n')
        return reconstructed


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, target):

        ind_half = int(outputs.size(0)/2)
       
        X_out = outputs[:ind_half]
        Y_out = outputs[ind_half:]

        X_target = target[:ind_half]
        Y_target = target[ind_half:]

        nsteps = ind_half
        loss = 0.0
        
        # Frobenius
        for step in range(nsteps):
            Xhat_X = torch.norm( X_target[step] - X_out[step], p='fro')
            Yhat_Y = torch.norm( Y_target[step] - Y_out[step], p='fro')
            loss += Xhat_X + Yhat_Y

        loss = loss/(nsteps*5.0)
        # L1
        # loss = np.mean(np.abs(outputs[:] - targets[:])/np.abs(targets[:]))

        return loss

class DataIO(object):
    def __init__(self,nst,nls,nin,gpaths,fpaths,iz):
        # Time steps
        self.nst, self.nls, self.nin = nst,nls,nin
        self.nstepall = np.arange(self.nst,self.nls,self.nin)
        # Read zone
        self.iz     = iz
        # Grid/Flow data
        self.gpaths = glob.glob(gpaths)
        self.fpath  = fpaths

    def readgrid(self):
        # Read grid files
        grids = []
        for gpath in self.gpaths:
            iheader = checkheader(gpath)
            grid, ibottom = readgrid(gpath,iheader,8)
            grids.append(grid)

        return grids,ibottom

    def readflow(self):
        # Read flow files
        flows = []
        for step in self.nstepall:
            fname = self.fpath+'flow_z{:0=5}_{:0=8}'.format(self.iz,step)
            iheader = checkheader(fname)
            q,self.statedic = readflow(fname,iheader)
            flows.append(q)

        return flows

    def writegrid(self,fnames,grids,jcuts,kcuts,lcuts):
        # settings
        jst,jls,jint = jcuts
        kst,kls,kint = kcuts
        lst,lls,lint = lcuts

        cjmax = jls-jst-jint+1 # cropped jmax
        ckmax = kls-kst-kint+1 # cropped kmax
        clmax = lls-lst-lint+1 # cropped lmax

        cjs,cje,cks,cke,cls,cle = 0,cjmax,0,ckmax,0,clmax
        ite1,ite2,jd,imove      = 0,0,0,0
        ibottom = cjs,cje,cks,cke,cls,cle,ite1,ite2,jd,imove

        for fname, grid in zip(fnames, grids):
            # tweak grid
            twgrid = grid[jst:jls:jint,kst:kls:kint,lst:lls:lint,:]

            # write grid files
            try:
                writegrid(fname,twgrid,4,ibottom,4)
            except:
                print("Check grid files header. (default = 4)")

    def writeflow(self,fname,q,jcuts,kcuts,lcuts):
        # settings
        jst,jls,jint = jcuts
        kst,kls,kint = kcuts
        lst,lls,lint = lcuts

        cjmax = jls-jst-jint+1 # cropped jmax
        ckmax = kls-kst-kint+1 # cropped kmax
        clmax = lls-lst-lint+1 # cropped lmax

        cjs,cje,cks,cke,cls,cle = 0,cjmax,0,ckmax,0,clmax
        ite1,ite2,jd,imove      = 0,0,0,0
        ibottom = cjs,cje,cks,cke,cls,cle,ite1,ite2,jd,imove

        fdata = np.zeros([cjmax,ckmax,clmax,5])

        # reshape
        fdata[:,0,:,0] = q[0,:,:]
        fdata[:,0,:,1] = q[1,:,:]
        fdata[:,0,:,2] = q[2,:,:]
        fdata[:,0,:,3] = q[3,:,:]
        fdata[:,0,:,4] = q[4,:,:]

        for i in range(1,ckmax):
            fdata[:,i,:,0] = fdata[:,0,:,0]
            fdata[:,i,:,1] = fdata[:,0,:,1]
            fdata[:,i,:,2] = fdata[:,0,:,2]
            fdata[:,i,:,3] = fdata[:,0,:,3]
            fdata[:,i,:,4] = fdata[:,0,:,4]

        # write flow files
        writeflow(fname,fdata,self.statedic,4)

class FlowDataset(torch.utils.data.Dataset):
    def __init__(self, ndim,jcuts,kcuts,lcuts,data,transform=None):
        self.transform = transform
        # Set data
        self.data = self.set_Data(ndim,jcuts,kcuts,lcuts,data)
        _,_,nlabels = self.data.shape
        self.labels = np.arange(nlabels)
        
        # cast
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        # length of data
        self.data_num = nlabels

    def __getitem__(self, idx):
        if self.transform:
            out_data = self.transform(self.data).view(self.data_num,5,self.cjmax,self.clmax)[idx]
            out_label = self.labels[idx]
        else:
            out_data = self.data.view(self.data_num,5,self.cjmax,self.clmax)[idx]
            out_label =  self.labels[idx]

        return out_data, out_label

    def __len__(self):
        return self.data_num


    def set_Data(self,ndim,jcuts,kcuts,lcuts,datas):
        jst,jls,jint = jcuts
        kst,kls,kint = kcuts
        lst,lls,lint = lcuts

        jmax,kmax,lmax,_ = datas[0].shape

        self.cjmax = jls-jst-jint+1 # cropped jmax
        self.ckmax = kls-kst-kint+1 # cropped kmax
        self.clmax = lls-lst-lint+1 # cropped lmax

        if ndim == 2:

            D = np.zeros([5,self.cjmax*self.clmax,len(datas)])
            print('Set D matrix ...')
            for i,data in enumerate(datas):
                qc =  data[jst:jls:jint,0,lst:lls:lint,0:5].reshape(self.cjmax*self.clmax,5)
                D[0,:,i] = qc[:,0] # rho
                D[1,:,i] = qc[:,1] # rho*u
                D[2,:,i] = qc[:,2] # rho*v
                D[3,:,i] = qc[:,3] # rho*w
                D[4,:,i] = qc[:,4] # energy

                D[2,:,:] = 0.0

            print('...... Finish \n')
        return D

class SlidingSampler(torch.utils.data.Sampler):
    def __init__(self,data,batch_size,sliding,shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.sliding = sliding
        self.indices = data[:][1]

        self.batches_indices = []
        max_index = max(self.indices) - 1
        start_ind = 0
        last_ind  = batch_size

        while last_ind <= max_index:
            self.batches_indices.append(list(np.arange(start_ind,last_ind)))
            start_ind = start_ind + self.sliding
            last_ind = start_ind + self.batch_size

        if shuffle==True:
            random.seed(0) # fix random number generator
            random.shuffle(self.batches_indices)

    def __iter__(self):

        return iter(self.batches_indices)
        
    def __len__(self):
        return self.batch_size

    def calc_shift_scale(self):
        input_last_indx = self.batches_indices[-1][-1]

        all_input = self.data[:input_last_indx][0]

        shift = torch.mean(all_input, (0,2,3)) # for each conservatives
        scale = torch.std(all_input,(0,2,3)) # for each conservatives

        return shift.to(torch.float32),scale.to(torch.float32)

        
