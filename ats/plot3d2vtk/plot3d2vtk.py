# -*- coding: utf-8 -*-
import numpy as np
from readwritePLOT3D import checkheader,readgrid,readflow,qtoqp,qptoq
from pyevtk.hl import gridToVTK
import re
import os
import sys
import configparser

from glob import glob
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0


class ROM:
    def __init__(self):
        if size < 2:
            print('MPI size too small. Please run np>=2',size, flush=True)

            sys.exit(1)

        # Extra arguments for checking files
        self.args = {}
        #self.args['check'] = None
        self.args['check'] = 'on'

        self.casesub = 'u'
#        self.allcases = ['u3.0','u4.0']
        self.allcases = ['u4.0']
        self.nst,self.nls,self.nin = 95100,114800,100
        self.nstepall=np.arange(self.nst,self.nls+self.nin,self.nin)        
        #  Cut region (start(j=1 is 0),end(jmax is None),interval)
        #  e.g., 1,jmax,jmax-2 <=> 0,None,-2
        self.jst,self.jls,self.jint = 0,None,1
        self.lst,self.lls,self.lint = 0,None,1
        self.kst,self.kls,self.kint = 0,None,1  # only k=1 plane
        self.ist,self.ils = 0,5 # default 0:rho, 1-3:uvw, 4:p
        self.iz = 3
        # self.datapath='/share/vol5/mimura/From_AFI/rod-valid-test/work/'
        self.datapath='../../recflows/'
        self.outpath='./'

    def run(self):
        if rank == root:   
            ana_list = []
            for casename in self.allcases:
                ana_list.append(casename)
        else:
            ana_list = None

        comm.barrier()
        if rank == root:
            for i in range(size-1):
                ana_list.append('STOP')
            for casename in ana_list:
                d = comm.recv(source=MPI.ANY_SOURCE)
                #print('send',casename,' to ',d)        
                comm.send(casename, dest=d)

        else:
            while True:
                comm.send(rank,root)
                casename = comm.recv(source=root)
                # print('receive',casename,' at ',rank)

                if casename == 'STOP':
                    print('Finish rank',rank, flush=True)
                    break
                else:
                    print('Start ',casename,' at rank',rank, flush=True)
                    failflag = False

                    # Read, cut, and write grid
                    fname=self.datapath+casename+'/grid_z'+'{:0=4}'.format(self.iz)

                    # Check header
                    if os.path.isfile(fname):
                        iheader = checkheader(fname)
                    else:
                        print('Grid does not exist... ',fname,' at rank',rank, flush=True)
                        failflag = True
                        break
                    print('Header:', iheader, flush=True)
                    
                    rall,idum = readgrid(fname,iheader,4)
                    rc = rall[self.jst:self.jls:self.jint,self.kst:self.kls:self.kint,self.lst:self.lls:self.lint,0:3]
                    jmax,kmax,lmax = rc.shape[:3]
                    print('   ...reading',fname,jmax,kmax,lmax)

                    # Read flow
                    flowpath=self.datapath+casename#+'/data/'
                    for k,nstep in enumerate(self.nstepall):
                        fname = flowpath+'/recflow_z'+'{:0=2}'.format(self.iz)+'_'+'{:0=8}'.format(nstep)

                        if np.mod(k,1) == 0:
                            print('rank',rank,' loading ',fname, flush=True)
                        try:
                            q,statedic = readflow(fname,iheader)
                        except:
                            print('### File not found ###: ',fname,flush=True)
                            print('            Skipping... ',casename,' at ',rank,flush=True)
                            failflag = True
                            break
                        qc = q[self.jst:self.jls:self.jint,self.kst:self.kls:self.kint,self.lst:self.lls:self.lint,0:5] # Cut region
                        qp = qtoqp(qc)[:,:,:,self.ist:self.ils]
                        print(qc.shape,rc.shape,flush=True)

                        x=np.array(rc[:,:,:,0],order = 'F')
                        y=np.array(rc[:,:,:,1],order = 'F')
                        z=np.array(rc[:,:,:,2],order = 'F')
                        rho=np.array(qp[:,:,:,0],order = 'F')
                        u=np.array(qp[:,:,:,1],order = 'F')
                        v=np.array(qp[:,:,:,2],order = 'F')
                        w=np.array(qp[:,:,:,3],order = 'F')
                        p=np.array(qp[:,:,:,4],order = 'F')

                        dname = self.outpath+'/'+casename
                        fname = 'recflow_z{0:0=2}_{1:0=8}'.format(self.iz,nstep)

                        os.makedirs(dname,exist_ok=True)
                        gridToVTK(dname+'/'+fname, x, y, z, 
                                  cellData = {}, 
                                  pointData = {"Density" : rho,
                                               "u" : u,
                                               "v" : v,
                                               "w" : w,
                                               "Pressure" : p})

def main():
    Cyl = ROM()
    Cyl.run()
    MPI.Finalize()    

if __name__ == '__main__':
    main()
