# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
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
            nn.Conv2d(256,128,stride=1,kernel_size=1,padding=1,bias=True)
        )
        self.eblock3 = nn.Sequential(
            enresidualblock(128,128)
        )        
        self.eblock34 = nn.Sequential(
            nn.Conv2d(128,64,stride=1,kernel_size=3,padding=1,bias=True)
        )
        self.eblock4 = nn.Sequential(
            enresidualblock(64,64)
        )
        self.eblock45 = nn.Sequential(
            nn.Conv2d(64,32,stride=1,kernel_size=3,padding=1,bias=True)
        )
        self.eblock5 = nn.Sequential(
            enresidualblock(32,32)
        )
        self.eblock56 = nn.Sequential(
            nn.Conv2d(32,16,stride=1,kernel_size=3,padding=1,bias=True)
        )
        self.eblock6 = nn.Sequential(
            enresidualblock(16,16)
        )
        self.efc = nn.Sequential(
            nn.Linear(fc_features,1)
        )
        #=== Encoder Block ===

        #=== Decoder Block ===
        deresidualblock = self.DecoderResidualBlock
        self.dfc = nn.Sequential(
            nn.Linear(1,fc_features)
        )
        self.dblock6 = nn.Sequential(
            enresidualblock(16,16)
        )
        self.dblock65 = nn.Sequential(
            nn.ConvTranspose2d(16,32,stride=1,kernel_size=3,padding=1,bias=True)
        )
        self.dblock5 = nn.Sequential(
            enresidualblock(32,32)
        )
        self.dblock54 = nn.Sequential(
            nn.ConvTranspose2d(32,64,stride=1,kernel_size=3,padding=1,bias=True)
        )
        self.dblock4 = nn.Sequential(
            enresidualblock(64,64)
        )
        self.dblock43 = nn.Sequential(
            nn.ConvTranspose2d(64,128,stride=1,kernel_size=3,padding=1,bias=True)
        )
        self.dblock3 = nn.Sequential(
            enresidualblock(128,128)
        )        
        self.dblock32= nn.Sequential(
            nn.ConvTranspose2d(128,256,stride=1,kernel_size=3,padding=1,bias=True)
        )
        self.dblock2 = nn.Sequential(
            enresidualblock(256,256)
        )
        self.dblock1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,5,stride=2,kernel_size=3,padding=3,bias=True)
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
        # Block 2
        # print('eblock2 ',x.size())
        x = self.eblock2(x)
        x = self.eblock23(x)
        # Block 3
        # print('eblock3 ',x.size())
        x = self.eblock3(x)
        x = self.eblock34(x)
        # Block 4
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
        x = x.view(-1)        
        x = self.efc(x)
        # print('Fully Connected',x.size())

        return x

    def decoderforward(self,x):
        # Fully Connected
        x = self.dfc(x)
        # print('Fully Conneted',x.size())
        x = x.view(self.eshape[0],self.eshape[1],self.eshape[2],self.eshape[3])
        # Block 4
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

    def forward(self, input_list):

        # print('Process... ')
        nsteps = len(input_list)

        # Encoder 
        enout = []
        for step in range(nsteps):
            enout.append(self.encoderforward(input_list[step]))

        # Decoder
        deout = []
        for step in range(nsteps):
            deout.append(self.decoderforward(enout[step]))

        reconstructed = deout
        # print('Finalze Process... \n')

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

    def forward(self, outputs, targets):

        nsteps = len(outputs)
        loss = 0.0
        
        # Frobenius
        for step in range(nsteps):
            loss += torch.norm(outputs[step] - targets[step],p='fro')

        loss = loss / nsteps

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
            fname = self.fpath+'flow_z{:0=2}_{:0=8}'.format(self.iz,step)
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

        fdata = np.zeros([cjmax,1,clmax,5])

        # reshape
        fdata[:,0,:,0] = q[0,:,:]
        fdata[:,0,:,1] = q[1,:,:]
        fdata[:,0,:,2] = q[2,:,:]
        fdata[:,0,:,3] = q[3,:,:]
        fdata[:,0,:,4] = q[4,:,:]

        # write flow files
        try:
            writeflow(fname,fdata,self.statedic,4)
        except:
            print("Check grid files header. (default = 4)")

class FlowDataset(torch.utils.data.Dataset):
    def __init__(self, ndim,jcuts,kcuts,lcuts,data,transform=None):
        self.transform = transform
        # Set data
        self.data = self.set_X(ndim,jcuts,kcuts,lcuts,data)
        _,_,nlabels = self.data.shape
        self.labels = np.arange(nlabels)
        
        # cast
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        # length of data
        self.data_num = nlabels


    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:

            out_data = self.transform(self.data).view(self.data_num,5,self.cjmax,self.clmax)[idx]
            out_label = self.labels[idx]
        else:
            out_data = self.data[idx]
            out_label =  self.labels[idx]

        return out_data, out_label

    def set_X(self,ndim,jcuts,kcuts,lcuts,datas):
        jst,jls,jint = jcuts
        kst,kls,kint = kcuts
        lst,lls,lint = lcuts

        jmax,kmax,lmax,_ = datas[0].shape

        self.cjmax = jls-jst-jint+1 # cropped jmax
        self.ckmax = kls-kst-kint+1 # cropped kmax
        self.clmax = lls-lst-lint+1 # cropped lmax

        if ndim == 2:

            D = np.zeros([5,self.cjmax*self.clmax,len(datas)])

            for i,data in enumerate(datas):
                qc =  data[jst:jls:jint,0,lst:lls:lint,0:5].reshape(self.cjmax*self.clmax,5)
                D[0,:,i] = qc[:,0] # rho
                D[1,:,i] = qc[:,1] # rho*u
                D[2,:,i] = qc[:,2] # rho*v
                D[3,:,i] = qc[:,3] # rho*w
                D[4,:,i] = qc[:,4] # energy

                D[2,:,:] = 0.0

        # Set X,Y
        X = D[:,:,0:-1]
        # Y = D[1::,:,:]

        return X
