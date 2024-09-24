#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:40:16 2024

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import filters

if __name__=='__main__':
    epsilon_=np.load('OverThrust.npy')
    epsilon=np.zeros((220,420))
    epsilon[10:-10,10:-10]=epsilon_
    CPML=10
    epsilon[:CPML,:]=epsilon[CPML,:]
    epsilon[-CPML:,:]=epsilon[-CPML-1,:]
    epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1))
    epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1))
    epsilon[:13,:]=1
    pyplot.figure()
    gci=pyplot.imshow(epsilon[10:-10,10:-10],extent=(0,2,1,0),cmap=cm.gray_r,vmin=1,vmax=8)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,2,4,6,8])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('00_TrueModel.png',dpi=1000)
    
    iepsilon=filters.gaussian(epsilon_,sigma=40)
    iepsilon[:3,:]=1
    pyplot.figure()
    gci=pyplot.imshow(iepsilon,extent=(0,2,1,0),cmap=cm.gray_r,vmin=1,vmax=8)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,2,4,6,8])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('01_InitModel.png',dpi=1000)
    
    data=np.load('400MHz_Forward_data.npy')
    pyplot.figure()
    gci=pyplot.imshow(data,extent=(0,2,1,0),cmap=cm.gray,vmin=-50,vmax=50)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,15,30,45,60])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (m)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_ticks([-50,-25,0,25,50])
    cbar.set_label('Amplitude')
    pyplot.savefig('02_400MHzForwardData.png',dpi=1000)
    
    data=np.load('./400000000.0Hz_imodel_file_38_4/4_imodel.npy')
    data=data.reshape((220,420))
    pyplot.figure()
    gci=pyplot.imshow(data[10:-10,10:-10],extent=(0,2,1,0),cmap=cm.gray_r,vmin=1,vmax=8)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,2,4,6,8])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('03_FWIModel.png',dpi=1000)
    
    idx=320
    ep_True=epsilon[10:-10,idx]
    ep_Init=iepsilon[:,idx]
    ep_FWI=data[10:-10,idx]
    pyplot.figure(figsize=(20,15))
    pyplot.axes().set_aspect(1.7e-1)
    l_a1=pyplot.plot(np.arange(len(ep_True))*0.02,ep_True,'b-',label='True Model')
    l_a2=pyplot.plot(np.arange(len(ep_True))*0.02,ep_Init,'k:',label='Initial Model')
    l_a3=pyplot.plot(np.arange(len(ep_True))*0.02,ep_FWI,'r-',label='Result of Comprehensive FWI Strategies')
    ax=pyplot.gca()
    ax.set_xlabel('Depth (m)', fontsize=16)
    ax.set_ylabel('$\epsilon_r$', fontsize=16) 
    lns=l_a1+l_a2+l_a3
    labs=[l.get_label() for l in lns]
    ax.legend(lns,labs,loc='best', fontsize=16)
    pyplot.grid(linestyle='--')
    pyplot.savefig('04_FWILine.png',dpi=1000)
    
    data1=np.load('Loss.npy')
    pyplot.figure(figsize=(12,10))
    pyplot.axes().set_aspect(3e-5)
    pyplot.plot(np.arange(len(data1))+1,data1,'b.-.',label='Ture Model')
    ax=pyplot.gca()
    ax.set_ylabel('Value of Objective Function', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12) 
    ax.ticklabel_format(style='sci',scilimits=(-1,2),axis='y')
    pyplot.grid(linestyle='--')
    pyplot.savefig('05_FWILoss.png',dpi=1000)
