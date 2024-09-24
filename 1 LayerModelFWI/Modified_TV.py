# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:09:22 2018

@author: Yilin Liu
"""
import numpy as np
from numba import jit

def denoising_2D_TV(data):
    data_max=np.max(data)
    data=data/(data_max+1e-6)
    M,N=np.shape(data)
    X0=np.zeros((M+2,N+2))
    X0[1: M+1,1: N+1]=data
    Y0=np.zeros((M+2,N+2))
    Y0[1: M+1,1: N+1]=data
    X=np.zeros((M+2,N+2))
    Zx=np.zeros((M+2,N+2))
    Zy=np.zeros((M+2,N+2))
    Ux=np.zeros((M+2,N+2))
    Uy=np.zeros((M+2,N+2))
    lamda=0.02
    rho_=1
    num=500
    err=1e-5
    return_data=denoising_2D_TV_(num,X,X0,err,M,N,Zx,Zy,Ux,Uy,Y0,lamda,rho_)
    return return_data*data_max

@jit(nopython=True)
def denoising_2D_TV_(num,X,X0,err,M,N,Zx,Zy,Ux,Uy,Y0,lamda,rho_):
    K=0
    while K<num and np.linalg.norm(X-X0,2) > err:
        # update X
        X0=X
        MM=M+2
        NN=N+2
        D=np.zeros((MM,NN))
        D[:,0:NN-1]=Zx[:,0:NN-1]-Zx[:,1:NN]
        D[:,NN-1]=Zx[:,NN-1]-Zx[:,0]
        Dxt_Zx=D
        
        D=np.zeros((MM,NN))
        D[0:MM-1,:]=Zy[0:MM-1,:]-Zy[1:MM,:]
        D[MM-1,:]=Zy[MM-1,:]-Zy[0,:]
        Dyt_Zy=D
        
        D=np.zeros((MM,NN))
        D[:,0:NN-1]=Ux[:,0:NN-1]-Ux[:,1:NN]
        D[:,NN-1]=Ux[:,NN-1]-Ux[:,0]
        Dxt_Ux=D
        
        D=np.zeros((MM,NN))
        D[0:MM-1,:]=Uy[0:MM-1,:]-Uy[1:MM,:]
        D[MM-1,:]=Uy[MM-1,:]-Uy[0,:]
        Dyt_Uy=D
        
        RHS=Y0+lamda*rho_*(Dxt_Zx+Dyt_Zy)-lamda*(Dxt_Ux+Dyt_Uy)
        X=np.zeros((M+2,N+2))
        
        for i in range(1,M+1):
            for j in range(1,N+1):
                X[i,j]=((X0[i+1,j]+X0[i-1,j]+X0[i,j+1]+X0[i,j-1])*lamda*rho_+RHS[i,j])/(1+4*lamda*rho_)
                
        # update Z
        D=np.zeros((MM,NN))
        D[:,1:NN]=X[:,1:NN]-X[:,0:NN-1]
        D[:,0]=X[:,0]-X[:,NN-1]
        Dx_X=D
        D=np.zeros((MM,NN))
        D[1:MM,:]=X[1:MM,:]-X[0:MM-1,:]
        D[0,:]=X[0,:]-X[MM-1,:]
        Dy_X=D
        Tx=Ux/rho_+Dx_X
        Ty=Uy/rho_+Dy_X
        
        Zx=np.fmax(np.fabs(Tx)-1/rho_,0)*np.sign(Tx)
        Zy=np.fmax(np.fabs(Ty)-1/rho_,0)*np.sign(Ty)
        
        # update U
        Ux=Ux+(Dx_X-Zx)
        Uy=Uy+(Dy_X-Zy)
        K+=1
    return X[1:M+1,1:N+1]