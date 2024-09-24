#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:29:03 2024

@author: nephilim
"""

from numba import jit
import numpy as np
from matplotlib import pyplot
from multiprocessing import Pool
import Wavelet
import Add_CPML
import time
import shutil
import os
import MultiScale

@jit(nopython=True)            
def update_H(xl,zl,dx,dz,dt,sigma,epsilon,mu,npml,a_x,a_z,b_x,b_z,k_x,k_z,Hz,Hx,Ey,memory_dEy_dx,memory_dEy_dz):
    x_len=xl+2*npml
    z_len=zl+2*npml

    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
            value_dEy_dx=(Ey[i+1][j]-Ey[i][j])/dx
                         
            if (i>=npml) and (i<x_len-npml):
                Hz[i][j]+=value_dEy_dx*dt/mu[i][j]
                
            elif i<npml:
                memory_dEy_dx[i][j]=b_x[i]*memory_dEy_dx[i][j]+a_x[i]*value_dEy_dx
                value_dEy_dx=value_dEy_dx/k_x[i]+memory_dEy_dx[i][j]
                Hz[i][j]+=value_dEy_dx*dt/mu[i][j]
                
            elif i>=xl-npml:
                memory_dEy_dx[i-xl][j]=b_x[i]*memory_dEy_dx[i-xl][j]+a_x[i]*value_dEy_dx
                value_dEy_dx=value_dEy_dx/k_x[i]+memory_dEy_dx[i-xl][j]
                Hz[i][j]+=value_dEy_dx*dt/mu[i][j]
                      
    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
            value_dEy_dz=(Ey[i][j+1]-Ey[i][j])/dz
                         
            if (j>=npml) and (j<z_len-npml):
                Hx[i][j]-=value_dEy_dz*dt/mu[i][j]
                
            elif j<npml:
                memory_dEy_dz[i][j]=b_z[j]*memory_dEy_dz[i][j]+a_z[j]*value_dEy_dz
                value_dEy_dz=value_dEy_dz/k_z[j]+memory_dEy_dz[i][j]
                Hx[i][j]-=value_dEy_dz*dt/mu[i][j]
                
            elif j>=z_len-npml:
                memory_dEy_dz[i][j-zl]=b_z[j]*memory_dEy_dz[i][j-zl]+a_z[j]*value_dEy_dz
                value_dEy_dz=value_dEy_dz/k_z[j]+memory_dEy_dz[i][j-zl]
                Hx[i][j]-=value_dEy_dz*dt/mu[i][j]

    return Hz,Hx

@jit(nopython=True)            
def update_E(xl,zl,dx,dz,dt,ca,cb,npml,a_x,a_z,b_x,b_z,k_x,k_z,Hz,Hx,Ey,memory_dHz_dx,memory_dHx_dz):
    x_len=xl+2*npml
    z_len=zl+2*npml
    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
            value_dv_dx=(Hz[i][j]-Hz[i-1][j])/dx
         
            value_dw_dz=(Hx[i][j]-Hx[i][j-1])/dz                        

            if (i>=npml) and (i<x_len-npml) and (j>=npml) and (j<z_len-npml):
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i<npml) and (j>=npml) and (j<z_len-npml):
                memory_dHz_dx[i][j]=b_x[i]*memory_dHz_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i][j]
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i>=x_len-npml) and (j>=npml) and (j<z_len-npml):
                memory_dHz_dx[i-xl][j]=b_x[i]*memory_dHz_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i-xl][j]
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (j<npml) and (i>=npml) and (i<x_len-npml):
                memory_dHx_dz[i][j]=b_z[j]*memory_dHx_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j]
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (j>=z_len-npml) and (i>=npml) and (i<x_len-npml):
                memory_dHx_dz[i][j-zl]=b_z[j]*memory_dHx_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j-zl]
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i<npml) and (j<npml):
                memory_dHz_dx[i][j]=b_x[i]*memory_dHz_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i][j]
                
                memory_dHx_dz[i][j]=b_z[j]*memory_dHx_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j]
                
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i<npml) and (j>=z_len-npml):
                memory_dHz_dx[i][j]=b_x[i]*memory_dHz_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i][j]
                
                memory_dHx_dz[i][j-zl]=b_z[j]*memory_dHx_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j-zl]
                
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i>=x_len-npml) and (j<npml):
                memory_dHz_dx[i-xl][j]=b_x[i]*memory_dHz_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i-xl][j]
                
                memory_dHx_dz[i][j]=b_z[j]*memory_dHx_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j]
               
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i>=x_len-npml) and (j>=z_len-npml):
                memory_dHz_dx[i-xl][j]=b_x[i]*memory_dHz_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i-xl][j]
                
                memory_dHx_dz[i][j-zl]=b_z[j]*memory_dHx_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j-zl]
                
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
    return Ey

#Forward modelling ------ timeloop
def time_loop(xl,zl,dx,dz,dt,sigma,epsilon,mu,CPML_Params,f,k_max,source_site,ref_pos):
    ep0 = 8.841941282883074e-12
    mu0 = 1.2566370614359173e-06
    epsilon *= ep0
    mu *= mu0
    npml=CPML_Params.npml        
    Ey=np.zeros((xl+2*npml,zl+2*npml))
    Hz=np.zeros((xl+2*npml,zl+2*npml))
    Hx=np.zeros((xl+2*npml,zl+2*npml))
        
    memory_dEy_dx=np.zeros((2*npml,zl+2*npml))
    memory_dEy_dz=np.zeros((xl+2*npml,2*npml))
    memory_dHz_dx=np.zeros((2*npml,zl+2*npml))
    memory_dHx_dz=np.zeros((xl+2*npml,2*npml))
    
    a_x=CPML_Params.a_x
    b_x=CPML_Params.b_x
    k_x=CPML_Params.k_x
    a_z=CPML_Params.a_z
    b_z=CPML_Params.b_z
    k_z=CPML_Params.k_z
    a_x_half=CPML_Params.a_x_half
    b_x_half=CPML_Params.b_x_half
    k_x_half=CPML_Params.k_x_half
    a_z_half=CPML_Params.a_z_half
    b_z_half=CPML_Params.b_z_half
    k_z_half=CPML_Params.k_z_half
    ca = CPML_Params.ca
    cb = CPML_Params.cb
            
    for tt in range(k_max):
        Hz,Hx=update_H(xl,zl,dx,dz,dt,ca,cb,mu,npml,a_x_half,a_z_half,b_x_half,b_z_half,k_x_half,k_z_half,Hz,Hx,Ey,memory_dEy_dx,memory_dEy_dz)
        Ey=update_E(xl,zl,dx,dz,dt,ca,cb,npml,a_x,a_z,b_x,b_z,k_x,k_z,Hz,Hx,Ey,memory_dHz_dx,memory_dHx_dz)
        Ey[source_site[0]][source_site[1]]+=-cb[source_site[0]][source_site[1]]*f[tt]*dt/dx/dz
        # pyplot.imshow(Ey,vmin=-50,vmax=50)
        # pyplot.pause(0.01)
        yield Ey[ref_pos[0],ref_pos[1]],

def Forward_2D(sigma,epsilon,mu,para):
    #Create Folder
    if not os.path.exists('./%sHz_forward_data_file'%para.Freq):
        os.makedirs('./%sHz_forward_data_file'%para.Freq)
    else:
        shutil.rmtree('./%sHz_forward_data_file'%para.Freq)
        os.makedirs('./%sHz_forward_data_file'%para.Freq)
    pool=Pool(processes=128)
    Profile=np.empty((para.k_max,len(para.source_site)))
    res_l=[]
    for index,data_position in enumerate(zip(para.source_site,para.receiver_site)):
        res=pool.apply_async(Forward2D,args=(para.xl,para.zl,para.Freq,para.k_max,sigma.copy(),epsilon.copy(),mu.copy(),para.dx,para.dz,para.dt,data_position[0],data_position[1],index,para.fs,para.target_freq))
        res_l.append(res)
    pool.close()
    pool.join()
    for res in res_l:
        result=res.get()
        Profile[:,result[0]]=result[1]
        del result
    del res_l
    pool.terminate() 
    np.save('./%sHz_forward_data_file/record.npy'%para.Freq,Profile)

def Forward2D(xl,zl,Freq,k_max,sigma,epsilon,mu,dx,dz,dt,value_source,value_receiver,index,fs,target_freq):
    t=np.arange(k_max)*dt
    f=Wavelet.ricker(t,Freq)
    
    f=MultiScale.apply_filter(f,fs,target_freq)
    
    CPML_Params=Add_CPML.Add_CPML(xl,zl,sigma.copy(),epsilon.copy(),mu.copy(),dx,dz,dt)
    Forward_data=time_loop(xl,zl,dx,dz,dt,sigma.copy(),epsilon.copy(),mu.copy(),CPML_Params,f,k_max,value_source,value_receiver)
    Profile=np.empty((k_max))
    for idx in range(k_max):
        tmp=Forward_data.__next__()
        Profile[idx]=tmp[0]
    return index,Profile
