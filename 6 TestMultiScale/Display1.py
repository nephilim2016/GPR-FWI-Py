#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:40:16 2024

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__=='__main__':
    epsilon_=np.load('LayerModel.npy')
    epsilon=np.zeros((120,220))
    epsilon[10:-10,10:-10]=epsilon_
    CPML=10
    epsilon[:CPML,:]=epsilon[CPML,:]
    epsilon[-CPML:,:]=epsilon[-CPML-1,:]
    epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1))
    epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1))
        
    epsilon_=epsilon.copy()
    epsilon[:13,:]=1
    pyplot.figure(figsize=(7.2,5))
    gci=pyplot.imshow(epsilon[10:-10,10:-10],extent=(0,2,1,0),cmap=cm.gray_r,vmin=1,vmax=10)
    # pyplot.savefig('00_Model.png',dpi=1000)
    
    data_400MHz=np.load('Add_Noise_Test01_400MHz_Original_data.npy')
    data_400MHz_Ricker_Filter=np.load('Add_Noise_Test02_400MHz_Original_data_Filter.npy')
    data_400MHz_Filter_Ricker=np.load('Add_Noise_Test04_400MHZ_Filter_Original_data.npy')
    t=np.arange(1000)*4e-11
    
    pyplot.figure(figsize=(7.2,5))
    gci=pyplot.imshow(data_400MHz,extent=(0,2,1,0),cmap=cm.gray,vmin=-50,vmax=50)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,0.5,1,1.5,2])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,10,20,30,40])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (ns)') 
    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('Amplitude')
    pyplot.savefig('Add_Noise_02_400MHz_Ricker_Data.png',dpi=1000)
    
    pyplot.figure(figsize=(7.2,5))
    gci=pyplot.imshow(data_400MHz_Ricker_Filter,extent=(0,2,1,0),cmap=cm.gray,vmin=-10,vmax=10)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,0.5,1,1.5,2])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,10,20,30,40])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (ns)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_ticks([-10,-5,0,5,10])
    cbar.set_label('Amplitude')
    pyplot.savefig('Add_Noise_03_400MHz_Ricker_Data_Filter.png',dpi=1000)
    
    pyplot.figure(figsize=(7.2,5))
    gci=pyplot.imshow(data_400MHz_Filter_Ricker,extent=(0,2,1,0),cmap=cm.gray,vmin=-10,vmax=10)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,0.5,1,1.5,2])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,10,20,30,40])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (ns)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_ticks([-10,-5,0,5,10])
    cbar.set_label('Amplitude')
    pyplot.savefig('Add_Noise_04_400MHz_Ricker_Filter_Data.png',dpi=1000)
    
    pyplot.figure(figsize=(7.2,5))
    gci=pyplot.imshow(data_400MHz_Ricker_Filter-data_400MHz_Filter_Ricker,extent=(0,2,1,0),cmap=cm.gray,vmin=-10,vmax=10)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,0.5,1,1.5,2])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,10,20,30,40])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (ns)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_ticks([-10,-5,0,5,10])
    cbar.set_label('Amplitude')
    pyplot.savefig('Add_Noise_05_Residual.png',dpi=1000)