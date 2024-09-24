#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:15:51 2024

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
import Wavelet
import MultiScale
from Optimization import para
import Forward2D
import AirForward2D

if __name__=='__main__':
    # Model Params
    para.xl=100
    para.zl=200
    para.dx=0.02
    para.dz=0.02
    para.k_max=1000
    para.dt=4e-11
    para.AirLayer=3
    # Ricker wavelet main frequence
    para.Freq=4e8
        
    para.MultiScale_Key=True
    para.fs=1/para.dt
    Target_freq=2e8
    
    # True Model
    epsilon_=np.load('LayerModel.npy')
    epsilon=np.zeros((para.xl+20,para.zl+20))
    epsilon[10:-10,10:-10]=epsilon_
    CPML=10
    epsilon[:CPML,:]=epsilon[CPML,:]
    epsilon[-CPML:,:]=epsilon[-CPML-1,:]
    epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1))
    epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1))
        
    epsilon_=epsilon.copy()
    epsilon[:10+para.AirLayer,:]=1
    
    sigma=np.ones((para.xl+20,para.zl+20))*1e-3
    sigma[:10+para.AirLayer,:]=0
    
    mu=np.ones((para.xl+20,para.zl+20))
    
    # Source Position
    para.source_site=[]
    para.receiver_site=[]
    for index in range(10,210,2):
        para.source_site.append((10,index))
        para.receiver_site.append((10,index))
            
    # Get True Model Data
    Forward2D.Forward_2D(sigma.copy(),epsilon.copy(),mu.copy(),para)
        
    # Save Profile Data
    para.Original_data=np.load('./%sHz_forward_data_file/record.npy'%para.Freq)
    para.Air_True_Profile_Original=AirForward2D.Forward_2D(np.zeros_like(sigma),np.ones_like(epsilon),np.ones_like(mu),para.Freq,para)
    para.Original_data=para.Original_data-np.tile(para.Air_True_Profile_Original,(para.Original_data.shape[1],1)).T

    para.data=np.zeros_like(para.Original_data)
    for idx_data_col in np.arange(para.data.shape[1]):
        para.data[:,idx_data_col]=MultiScale.apply_filter(para.Original_data[:,idx_data_col],para.fs,Target_freq)
    para.Air_True_Profile=MultiScale.apply_filter(para.Air_True_Profile_Original,para.fs,Target_freq)
    
    pyplot.figure()
    t=np.arange(para.k_max)*para.dt
    f=Wavelet.ricker(t,para.Freq)
    pyplot.plot(t,f)
    ax=pyplot.gca()
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude') 
    ax.set_aspect(1e-8)
    np.save('Test00_400MHz_Ricker.npy',f)
    
    pyplot.figure()
    gci=pyplot.imshow(para.Original_data,extent=(0,2,1,0),cmap=cm.gray,vmin=-50,vmax=50)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,0.5,1,1.5,2])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,10,20,30,40])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (ns)') 
    np.save('Test01_400MHz_Original_data.npy',para.Original_data)
    
    pyplot.figure()
    gci=pyplot.imshow(para.data,extent=(0,2,1,0),cmap=cm.gray,vmin=-50,vmax=50)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,0.5,1,1.5,2])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,10,20,30,40])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (ns)') 
    np.save('Test02_400MHz_Original_data_Filter.npy',para.data)