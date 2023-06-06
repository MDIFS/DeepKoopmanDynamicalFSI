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

class FSI(object):
    def __init__(self,jcuts,kcuts,lcuts,iz,dataio,mach,re,dt,inptype,ured,horizon):
        self.gpath = 'grid_z0003_r8k3'

        self.jcuts,self.kcuts,self.lcuts = jcuts,kcuts,lcuts
        self.iz = iz
        self.dataio = dataio
        self.mach,self.re,self.dt = mach,re,dt
        self.inptype = inptype
        self.ured = ured
        self.horizon = horizon

    def calc_force(self,X,u):
        X_prd             = X

        gpath             = self.gpath
        jcuts,kcuts,lcuts = self.jcuts,self.kcuts,self.lcuts
        iz                = self.iz
        dataio            = self.dataio
        mach,re,dt        = self.mach,self.re,self.dt
        inptype           = self.inptype
        ured              = self.ured

        forc_fluid = self.fluid_force(gpath,X_prd,jcuts,kcuts,lcuts,iz,dataio,mach,re,dt\
)
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
        print('MPC : horizontal window : T = ',self.T,'\n')
        
        if inptype==8:
            self.cndim = 2
    

    def solve_cvx(self,forces,Rval,A,B):
        T    = self.T
        Amat = A.to('cpu').detach().numpy().copy()
        Bmat = B.to('cpu').detach().numpy().copy()
        fvec = torch.squeeze(forces[0])[None].to('cpu').detach().numpy().copy() # fluid forces
        lamvec = torch.squeeze(forces[1])[None].to('cpu').detach().numpy().copy() # structure forces

        xndim = T
        cndim = self.cndim

        # Q = np.identity(xndim)
        Q = 1.0
        R = Rval*np.eye(cndim)

        # Set variables
        x = cvx.Variable(shape=(1,T))
        u = cvx.Variable(shape=(cndim,T-1))

        # set optimization problem
        cost = 0.0
        constr = []
        print(Amat)
        print(Bmat)
        exit()
        for t in range(T-1):
            # cost +=  (x[0,t+1] - lamvec[0,t+1])**2 + cvx.quad_form(u[:,t], R) # For Matrix Q
            # cost += (x[0,t+1] - lamvec[0,t+1])**2 #+ cvx.quad_form(u[:,t], R) # For Matrix Q
            cost += (x[0,t+1] - lamvec[0,t+1])**2 + cvx.quad_form(u[:,t],R) # For Matrix Q
            constr += [ x[0,t+1] == Amat*x[0,t]+Bmat@u[:,t] ]
        # solve
        obj  = cvx.Minimize(cost)
        prob = cvx.Problem(obj,constr)
        prob.solve(verbose=True)
        print(u.value)
        exit()
        return
