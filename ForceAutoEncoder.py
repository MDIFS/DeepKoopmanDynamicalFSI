# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import configparser
from distutils.util import strtobool
import copy
import subprocess
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.sampler
import torchvision

import cvxpy as cvx

from readwritePLOT3D import checkheader, readgrid, writegrid, \
                                         readflow, writeflow
 
class FAE(nn.Module):
    def __init__(self):
        super().__init__()
        setup = configparser.ConfigParser()
        setup.read('input.ini')
        fc_features = int(setup['DeepLearning']['full_connected'])
        window_size = int(setup['DeepLearning']['batchsize'])
        self.regression = float(setup['DeepLearning_force']['regression'])
        self.control     = strtobool(setup['Control']['control'])
        if self.control: 
            inptype = int(setup['Control']['inptype']) 
            self.Bf  = self.build_Bmat(fc_features,inptype)

        #=== Encoder  ===
        self.eblock1 = nn.Sequential(
            nn.Conv1d(1,256,kernel_size=3)

        )
        self.eblock2 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256,128,kernel_size=3)
        )
        self.eblock3 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128,64,kernel_size=3)
        )
        self.eblock4 = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64,32,kernel_size=3)
        )
        # self.eblock5 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(32),
        #     nn.Conv1d(32,16,kernel_size=3)
        # )
        self.efc = nn.Sequential(
            nn.Linear(800,window_size)
        )
        #=== Encoder Block ===

        #=== Decoder Block ===
        self.dfc = nn.Sequential(
            nn.Linear(fc_features,800)
        )
        # self.dblock5 = nn.Sequential(
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(True),
        #     nn.ConvTranspose1d(32,64,kernel_size=5)
        # )
        self.dblock4 = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32,64,kernel_size=5)
        )
        self.dblock3 = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64,128,kernel_size=5)
        )
        self.dblock2 = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128,256,kernel_size=5)
        )
        self.dblock1 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256,1,kernel_size=3)
        )
        #=== Decoder Block ===


    def encoderforward(self,x):
        # Block 1
        # print('eblock1 ',x.size())
        x = self.eblock1(x)
        # Block 2
        # print('eblock2 ',x.size())
        x = self.eblock2(x)
        # Block 3
        # print('eblock3 ',x.size())
        x = self.eblock3(x)
        # Block 4
        # print('eblock4 ',x.size())
        x = self.eblock4(x)
        # Block 5
        # print('eblock5 ',x.size())
        # x = self.eblock5(x)
        # Fully Connected
        x = x.view(1,-1)
        # print('viewed',x.size())
        x = self.efc(x)

        return x

    def decoderforward(self,x):
        # Fully Connected
        # print('decoder :',x.size())
        x = self.dfc(x)
        # print('Fully Conneted',x.size())
        # Block 5
        # print('dblock5 ',x.size())
        # x = self.dblock5(x)
        # Block 4
        x = x.view(32,-1)
        x = x[None]
        # print('dblock4 ',x.size())
        x = self.dblock4(x)
        # Block 3
        # print('dblock3 ',x.size())
        x = self.dblock3(x)
        # Block 2
        # print('dblock2 ',x.size())
        x = self.dblock2(x)
        # Block 1
        # print('dblock1 ',x.size())
        x = self.dblock1(x)
        # Fin
        # print('Final   ',x.size())
        # exit()
        x = torch.squeeze(x)
        return x

    def calc_Ytilde_prd(self,inp,u=0.0):
        # print('inp size=',inp.size())
        inp    = torch.squeeze(inp)
        Xtilde = inp[:-1]
        Ytilde = inp[1:]

        if self.control:
            Ytilde = Ytilde - u @ self.Bf

        # add axis
        Xtilde = Xtilde[None].T
        Ytilde = Ytilde[None].T

        # least-square with regularization
        m,n = Xtilde.size()
        l2_lambda = torch.tensor(self.regression)

        # regulalized least-square
        # Af is transposed
        # print('Xtilde',Xtilde)
        # print('Ytilde',Ytilde)
        if m < n:
            # print('minimum norm =',m,n)
            Af = torch.t(Xtilde.double()) @ torch.linalg.pinv( Xtilde.double() @ torch.t(Xtilde.double()) + l2_lambda.double()) @ Ytilde.double()
        else:
            # print('least square =',m,n)
            Af = torch.linalg.pinv( torch.t(Xtilde.double()) @ Xtilde.double() + l2_lambda.double()) @ torch.t(Xtilde.double()) @ Ytilde.double()
        
        # normal least-square
        # Af = torch.linalg.lstsq(Xtilde.to(torch.float32),Ytilde.to(torch.float32)).solution
        # koopman
        # Ytilde_pred = Xtilde.to(torch.float32) @ Af.to(torch.float32)

        # recursive
        Ytilde_pred  = torch.zeros( Ytilde.size() ).to('cuda')
        Ytilde_pred_i = Xtilde[0,0]
        # print('Xtilde',Xtilde.size(),Xtilde[0,0])
        # print('Af=',Af.size(),Af)
        Af_i = Af.double()
        # print('Ytilde=',Ytilde.size())
        # print('Ytilde_pred=',Ytilde_pred.size())
        for i in range(Ytilde.size(0)):
            Ytilde_pred[i,0] = Ytilde_pred_i
            Ytilde_pred_i =  Ytilde_pred_i.double() * Af_i

        # out = torch.cat((Xtilde, Ytilde_pred), axis = 0).half()
        # out = torch.cat((Xtilde, Ytilde_pred), axis = 0)

        out = torch.cat((Xtilde.T, Ytilde_pred.T), axis = 0)
        # exit()
        # Without A-matrix
        # out = torch.cat((Xtilde, Ytilde), axis =0)

        # print('Af=',Af,Af.size(),'\n')
        # print('Xtilde=',Xtilde,Xtilde.size(),'\n')
        # print('Ytilde_pred=',Ytilde_pred,Ytilde_pred.size(),'\n')
        # print('Ytilde=',Ytilde,Ytilde.size(),'\n')

        return out, Af

    def build_Bmat(self,fc_features,inptype):
        if inptype in [1,2,4,5,9]:
            inpdim = 1
        elif inptype in [6,7,8]:
            inpdim = 2
        elif inptype in [3]:
            inpdim = 3
        else:
            print('inptype is wrong. change 1~9')
            exit()

        B = nn.Parameter( nn.init.xavier_uniform_( torch.empty((inpdim, fc_features))) )

        return B
        
    def forward(self, inplist):
        inp = inplist[0]

        if self.control: u = inplist[1]

        # Encoder
        # print('Encoder')
        enout = self.encoderforward(inp)

        # least square
        # print('Least square')
        if self.control:
            enout,_ = self.calc_Ytilde_prd(enout,u)
        else:
            enout,_ = self.calc_Ytilde_prd(enout)

        # Decoder
        # print('Decoder')
        deout = self.decoderforward(enout)
        
        reconstructed = deout

        return reconstructed

    def forward_MPC_A_Bmat_X_Y_hat(self, inplist):
        inp = inplist[0]
        if self.control: u = inplist[1][1:]

        # Encoder
        # print('Encoder')
        enout = self.encoderforward(inp)

        # least square
        # print('Least square')
        if self.control:
            enout,A = self.calc_Ytilde_prd(enout,u)
        else:
            enout,A = self.calc_Ytilde_prd(enout)

        # Decoder
        # print('Decoder')
        deout = self.decoderforward(enout)

        reconstructed = deout

        return reconstructed, A.T, self.B.T

    def encoder_forMPC(self, inplist):
        inp = inplist[0]
        if self.control: u = inplist[1]

        # Encoder
        # print('Encoder')
        enout = self.encoderforward(inp)

        # least square
        # print('Least square')
        if self.control:
            enout,A = self.calc_Ytilde_prd(enout,u)
        else:
            enout,A = self.calc_Ytilde_prd(enout)

        return enout.T, A.T, self.Bf.T


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

        loss = loss/nsteps

        # L1
        # loss = np.mean(np.abs(outputs[:] - targets[:])/np.abs(targets[:]))

        return loss

class DataIO(object):
    def __init__(self,nst,nls,nin,gpaths,fpaths,iz,fmpaths=False):
        # Time steps
        self.nst, self.nls, self.nin = nst,nls,nin
        self.nstepall = np.arange(self.nst,self.nls,self.nin)
        # Read zone
        self.iz     = iz
        # Grid/Flow data
        self.gpaths = glob.glob(gpaths)
        self.fpath  = fpaths
        self.fmpath = fmpaths

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

    def tweak_writegrid(self,fnames,grids,jcuts,kcuts,lcuts):
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
                writegrid(fname+'_r8',twgrid,4,ibottom,8)

                if twgrid.shape[1] == 1:
                    rk3 = np.zeros((twgrid.shape[0],3,twgrid.shape[2],3))
                    for k in range(3):
                        rk3[:,k,:,:] = twgrid[:,0,:,:] + 0.5*k
                    writegrid(fname+'_r8k3',rk3,4,ibottom,8)
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

    def tweak_writeflow(self,fname,q,jcuts,kcuts,lcuts):
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

        fdata = np.zeros([cjmax,3,clmax,5])

        # reshape
        fdata[:,0,:,0] = q[0,:,:]
        fdata[:,0,:,1] = q[1,:,:]
        fdata[:,0,:,2] = q[2,:,:]
        fdata[:,0,:,3] = q[3,:,:]
        fdata[:,0,:,4] = q[4,:,:]

        for i in range(1,3):
            fdata[:,i,:,0] = fdata[:,0,:,0]
            fdata[:,i,:,1] = fdata[:,0,:,1]
            fdata[:,i,:,2] = fdata[:,0,:,2]
            fdata[:,i,:,3] = fdata[:,0,:,3]
            fdata[:,i,:,4] = fdata[:,0,:,4]

        # write flow files
        writeflow(fname,fdata,self.statedic,4)


    def readformom(self,inptype):
        formoms = np.loadtxt(self.fmpath)
        nsteps  = formoms[:,0]
        nind    = [np.where(nsteps == nstep)[0][0] for nstep in self.nstepall] 
        if len(nind) != len(self.nstepall):
            print('nstep of formom does not match')
            exit()

        osciz = formoms[nind,1].reshape(1,-1)
        dosciz = formoms[nind,2].reshape(1,-1)
        ddosciz = formoms[nind,3].reshape(1,-1)
        cl = formoms[nind,7].reshape(1,-1)

        if inptype == 0:
            u = cl
        elif inptype == 1:
            u = osciz
        elif inptype == 2:
            u = np.diff(osciz,prepend=0.0)
        elif inptype == 3:
            u = np.vstack((np.vstack((osciz,dosciz)),ddosciz))
        elif inptype == 4:
            u = dosciz
        elif inptype == 5:
            u = ddosciz
        elif inptype == 6:
            u = np.vstack((osciz,dosciz))
        elif inptype == 7:
            u = np.vstack((dosciz,ddosciz))
        elif inptype == 8:
            u = np.vstack((osciz,ddosciz))
        elif inptype == 9:
            u = np.ones_like(self.ddosciz)

        return u

class ForceDataset(torch.utils.data.Dataset):
    def __init__(self, ndim,jcuts,kcuts,lcuts,data,window_size,sliding,
                 control_inp=None,control=False,transform=None):

        self.control   = control

        self.transform = transform
        # Set data/labels
        self.data,self.labels = self.setdata(data,window_size,sliding)

        if self.control:
            control_tmp = control_inp[:,self.labels]
            self.diminp = control_tmp.shape[0]
            self.control_inp = np.expand_dims(control_tmp,axis=0)

        # length of data
        self.data_num = self.data.shape[2]

        # cast
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def setdata(self,data,window_size,sliding):
        indices = []
        labels = []
        max_index = data.shape[1]
        start_ind = 0
        last_ind = window_size
        
        while last_ind <= max_index:
            indices.append(list(np.arange(start_ind,last_ind)))
            labels.append(start_ind)
            start_ind = start_ind + sliding
            last_ind = start_ind + window_size

        inpdata_tmp = np.zeros((len(indices),1,window_size))

        for i in range(len(indices)):
            inpdata_tmp[:][i] = data[0][indices[i]]
            
        # rehsape (C H W) to (W H C)  
        inpdata = inpdata_tmp.transpose(2,1,0)

        print('... Finish \n')
        return inpdata,labels

    def __getitem__(self, idx,out_input=0.0):
        if self.transform:
            out_data = self.transform(self.data)[idx].view(1,-1)
            out_label = self.labels[idx]
            if self.control:out_input = self.transform(self.control_inp).view(-1,self.diminp)[idx]
        else:
            out_data = self.data.view(self.data_num,1,-1)[idx]
            out_label =  self.labels[idx]
            if self.control:out_input = self.control_inp.view(-1,self.diminp)[idx]

        return out_data, out_label, out_input

    def __len__(self):
        return self.data_num


class SlidingSampler(torch.utils.data.Sampler):
    def __init__(self,data,batch_size,sliding,shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.sliding = sliding
        self.batches_indices = np.arange(data[:][1].size()[0])

        if shuffle==True:
            random.seed(1) # fix random number generator
            random.shuffle(self.batches_indices)

    def __iter__(self):
        return iter(self.batches_indices)
        
    def __len__(self):
        return self.batch_size

    def calc_shift_scale(self,data):
        all_input = data

        shift = torch.mean(all_input) # for each conservatives
        scale = torch.std(all_input) # for each conservatives
    
        return shift.to(torch.float32),scale.to(torch.float32)

    def calc_min_max(self,data):
        all_input = data

        fmin = torch.min(all_input) # for each conservatives
        fmax = torch.max(all_input) # for each conservatives
    
        return fmin.to(torch.float32),fmax.to(torch.float32)

        
class FSI(object):
    def __init__(self,jcuts,kcuts,lcuts,iz,dataio,mach,re,dt,inptype,ured):
        self.gpath = 'grid_z0003_r8k3'
        
        self.jcuts,self.kcuts,self.lcuts = jcuts,kcuts,lcuts
        self.iz = iz
        self.dataio = dataio
        self.mach,self.re,self.dt = mach,re,dt
        self.inptype = inptype
        self.ured = ured
        
    def calc_force(self,X,u):
        X_prd             = X

        gpath             = self.gpath
        jcuts,kcuts,lcuts = self.jcuts,self.kcuts,self.lcuts
        iz                = self.iz
        dataio            = self.dataio
        mach,re,dt        = self.mach,self.re,self.dt
        inptype           = self.inptype
        ured              = self.ured

        forc_fluid = self.fluid_force(gpath,X_prd,jcuts,kcuts,lcuts,iz,dataio,mach,re,dt)
        forc_struct = self.structure_force(u,inptype,ured,mach) 

        return forc_fluid,forc_struct

    def fluid_force(self, gpath,X_prd,jcuts,kcuts,lcuts,iz,dataio,mach,re,dt):
        forc = np.zeros((1,X_prd.size(0)))
 
        for t in range(X_prd.size(0)):
            # output flow
            fname = './calcforce/t{:0=4}'.format(t)
            q = X_prd[t].cpu().numpy()
            dataio.tweak_writeflow(fname,q,jcuts,kcuts,lcuts)

            # calcforce
            f = open('./calcforce/stdin','w')
            lines = [str(mach)+'\n', str(re)+'\n', str(dt)+'\n', gpath+'\n', fname+'\n']
            f.writelines(lines)
            f.close()
            subprocess.call(['./calcforce/calcforce.out'])
            data = np.loadtxt('./calcforce/liftforce.dat',delimiter=',')

            forc[0,t] = data[0] # CL

        return forc

    def structure_force(self,u,inptype,ured,mach):
        forc = np.zeros((1,u.size(0)))

        Fs = 1.0 / ured
        xi = 0.0                       # Damping coef
        dmflu = 0.5                    # Coef of Cl
        dmm   = (2.0*np.pi*Fs)**2      # coef of Z (Normalized Spring coef)
        dmdamp = 4.0 * xi * np.pi * Fs # coef of Z'(Normalized Dampling coef)

        for t in range(u.size(0)):
            if inptype == 8:
                z   = u[t,0]
                dz  = 0.0
                ddz = u[t,-1] 

            forc[0,t] = (ddz + dmdamp*dz + dmm*z)/dmflu
        
        return forc

class ConvxOpt(object):
    def __init__(self,batch_size,inptype):

        self.T = int( (batch_size - 1)/2 )
        print('MPC : T = ',self.T,'\n')
        
        if inptype==8:
            self.cndim = 2
    

    def solve_cvx(self,forces,Rval):
        T    = self.T
        fvec = forces[0]   # fluid forces
        lamvec = forces[1] # structure forces

        xndim = fvec.shape[0]
        cndim = self.cndim

        Q = np.eye(xndim)
        R = Rval*np.eye(cndim)

        # Set variables
        u = cvx.Variable(shape=(cndim,T-1))

        # set optimization problem
        cost = 0.0
        constr = []
        for t in range(T-1):
        #cost += cvx.quad_form((lamvec[0,t] - fvec[0,t]), Q ) + cvx.quad_form(u[:,t],R) # For Matrix Q
            cost += (lamvec[0,t]-fvec[0,t])**2 + cvx.quad_form(u[:,t],R) # For Matrix Q
            constr += [lamvec[0,t]*u[0,t] == fvec[0,t]*u[0,t]]
        # solve
        obj  = cvx.Minimize(cost)
        prob = cvx.Problem(obj,constr)
        prob.solve(verbose=True)
        print(u.value)
        exit()
        return
