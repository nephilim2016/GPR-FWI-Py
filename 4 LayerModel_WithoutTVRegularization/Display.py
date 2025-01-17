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
    pyplot.figure()
    gci=pyplot.imshow(epsilon[10:-10,10:-10],extent=(0,2,1,0),cmap=cm.gray_r,vmin=1,vmax=10)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,0.5,1,1.5,2])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$\epsilon_r$')
    # pyplot.savefig('00_TrueModel.png',dpi=1000)
    
    CPML=10
    iep=np.linspace(1,10,100)
    iep=np.tile(iep,(200,1)).T
    iepsilon=np.zeros((120,220))
    iepsilon[10:-10,10:-10]=iep
    iepsilon[:CPML,:]=iepsilon[CPML,:]
    iepsilon[-CPML:,:]=iepsilon[-CPML-1,:]
    iepsilon[:,:CPML]=iepsilon[:,CPML].reshape((len(iepsilon[:,CPML]),-1))
    iepsilon[:,-CPML:]=iepsilon[:,-CPML-1].reshape((len(iepsilon[:,-CPML-1]),-1))
    iepsilon[:13,:]=1
    pyplot.figure()
    gci=pyplot.imshow(iepsilon[10:-10,10:-10],extent=(0,2,1,0),cmap=cm.gray_r,vmin=1,vmax=10)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,0.5,1,1.5,2])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$\epsilon_r$')
    # pyplot.savefig('01_InitModel.png',dpi=1000)
    
    data=np.load('./400000000.0Hz_imodel_file_22_4/4_imodel.npy')
    data=data.reshape((120,220))
    pyplot.figure()
    gci=pyplot.imshow(data[10:-10,10:-10],extent=(0,2,1,0),cmap=cm.gray_r,vmin=1,vmax=10)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,2,5))
    ax.set_xticklabels([0,1,2,3,4])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0,0.5,1,1.5,2])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('03_FWIModel.png',dpi=1000)
    
    # idx=60
    # ep_True=epsilon[10:-10,idx]
    # ep_Init=iepsilon[10:-10,idx]
    # ep_FWI=data[10:-10,idx]
    # pyplot.figure(figsize=(10,8))
    # pyplot.axes().set_aspect(8)
    # l_a1=pyplot.plot(ep_True,np.arange(len(ep_True))*0.02,'b-.',label='Ture Model')
    # l_a2=pyplot.plot(ep_Init,np.arange(len(ep_True))*0.02,'k:',label='Initial Model')
    # l_a3=pyplot.plot(ep_FWI,np.arange(len(ep_True))*0.02,'r-',label='FWI result')
    # pyplot.gca().invert_yaxis()
    # ax=pyplot.gca()
    # ax.set_ylabel('Depth (m)', fontsize=12)
    # ax.set_xlabel('$\epsilon_r$', fontsize=12) 
    # lns=l_a1+l_a2+l_a3
    # labs=[l.get_label() for l in lns]
    # ax.legend(lns,labs,loc='best')
    # pyplot.grid(linestyle='--')
    # pyplot.savefig('04_FWILine.png',dpi=1000)
    
    data=np.load('Loss.npy')
    pyplot.figure(figsize=(12,10))
    pyplot.axes().set_aspect(2.5e-4)
    pyplot.plot(np.arange(len(data))+1,data,'b.-.',label='Ture Model')
    ax=pyplot.gca()
    ax.set_ylabel('Value of Objective Function', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12) 
    ax.ticklabel_format(style='sci',scilimits=(-1,2),axis='y')
    pyplot.grid(linestyle='--')
    pyplot.savefig('05_FWILoss.png',dpi=1000)
    
    data=np.load('./400000000.0Hz_imodel_file_22_4/4_imodel.npy')
    data=data.reshape((120,220))
    pyplot.figure()
    gci=pyplot.imshow(data[35:85,135:185],extent=(0,1,1,0),cmap=cm.jet,vmin=3,vmax=8)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,1,5))
    ax.set_xticklabels([2.5,2.75,3,3.25,3.5])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([0.5,0.75,1,1.25,1.5])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)') 
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='5%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$\epsilon_r$')
    pyplot.savefig('Add_06_FWIModelPart.png',dpi=1000)