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
    epsilon_=np.load('./1 LayerModelFWI/LayerModel.npy')
    epsilon=np.zeros((120,220))
    epsilon[10:-10,10:-10]=epsilon_
    CPML=10
    epsilon[:CPML,:]=epsilon[CPML,:]
    epsilon[-CPML:,:]=epsilon[-CPML-1,:]
    epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1))
    epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1))
    epsilon_=epsilon.copy()
    epsilon[:13,:]=1

    iep=np.linspace(1,10,100)
    iep=np.tile(iep,(200,1)).T
    iepsilon=np.zeros((120,220))
    iepsilon[10:-10,10:-10]=iep
    iepsilon[:CPML,:]=iepsilon[CPML,:]
    iepsilon[-CPML:,:]=iepsilon[-CPML-1,:]
    iepsilon[:,:CPML]=iepsilon[:,CPML].reshape((len(iepsilon[:,CPML]),-1))
    iepsilon[:,-CPML:]=iepsilon[:,-CPML-1].reshape((len(iepsilon[:,-CPML-1]),-1))
    iepsilon[:13,:]=1

    data1=np.load('./1 LayerModelFWI/400000000.0Hz_imodel_file_22_0/4_imodel.npy')
    data1=data1.reshape((120,220))
    
    data2=np.load('./2 LayerModel_WithoutMultiply/400000000.0Hz_imodel_file_22_4/4_imodel.npy')
    data2=data2.reshape((120,220))
    
    data3=np.load('./3 LayerModel_WithoutRandomSource/400000000.0Hz_imodel_file_100_0/19_imodel.npy')
    data3=data3.reshape((120,220))
    
    data4=np.load('./4 LayerModel_WithoutTVRegularization/400000000.0Hz_imodel_file_22_4/4_imodel.npy')
    data4=data4.reshape((120,220))
    
    idx=50+10
    ep_True=epsilon[10:-10,idx]
    ep_Init=iepsilon[10:-10,idx]
    ep_FWI1=data1[10:-10,idx]
    ep_FWI2=data2[10:-10,idx]
    ep_FWI3=data3[10:-10,idx]
    ep_FWI4=data4[10:-10,idx]
    pyplot.figure(figsize=(20,15))
    pyplot.axes().set_aspect(8e-2)
    l_a1=pyplot.plot(np.arange(len(ep_True))*0.02,ep_True,'b-',label='True Model')
    l_a2=pyplot.plot(np.arange(len(ep_True))*0.02,ep_Init,'k:',label='Initial Model')
    l_a3=pyplot.plot(np.arange(len(ep_True))*0.02,ep_FWI1,'r-',label='Result of Comprehensive FWI (S1) Strategies')
    l_a4=pyplot.plot(np.arange(len(ep_True))*0.02,ep_FWI2,'c-.',label='Result of FWI without Multi-scale (S2) Strategy')
    l_a5=pyplot.plot(np.arange(len(ep_True))*0.02,ep_FWI3,'y:',label='Result of FWI without Random Excitation Source (S3) Strategy')
    l_a6=pyplot.plot(np.arange(len(ep_True))*0.02,ep_FWI4,'m-.',label='Result of FWI without TV Regularization (S4) Strategy')
    ax=pyplot.gca()
    ax.set_xlabel('Depth (m)', fontsize=16)
    ax.set_ylabel('$\epsilon_r$', fontsize=16) 
    lns=l_a1+l_a2+l_a3+l_a4+l_a5+l_a6
    labs=[l.get_label() for l in lns]
    ax.legend(lns,labs,loc='best', fontsize=16)
    pyplot.grid(linestyle='--')
    pyplot.savefig('04_FWILine.png',dpi=1000)

    pyplot.figure()
    pyplot.axes().set_aspect(2e-1)
    l_a1=pyplot.plot(np.arange(len(ep_True))[28:50]*0.02,ep_True[28:50],'b-',label='True Model')
    l_a2=pyplot.plot(np.arange(len(ep_True))[28:50]*0.02,ep_Init[28:50],'k:',label='Initial Model')
    l_a3=pyplot.plot(np.arange(len(ep_True))[28:50]*0.02,ep_FWI1[28:50],'r-',label='Result of Comprehensive FWI Strategies')
    l_a4=pyplot.plot(np.arange(len(ep_True))[28:50]*0.02,ep_FWI2[28:50],'c-.',label='Result of FWI without Multi-scale Strategy')
    l_a5=pyplot.plot(np.arange(len(ep_True))[28:50]*0.02,ep_FWI3[28:50],'y:',label='Result of FWI without Random Excitation Source Strategy')
    l_a6=pyplot.plot(np.arange(len(ep_True))[28:50]*0.02,ep_FWI4[28:50],'m-.',label='Result of FWI without TV Regularization Strategy')
    pyplot.ylim([4,5])
    pyplot.savefig('05_FWILine0.png',dpi=1000)
    
    pyplot.figure()
    pyplot.axes().set_aspect(0.5e-1)
    l_a1=pyplot.plot(np.arange(len(ep_True))[55:72]*0.02,ep_True[55:72],'b-',label='True Model')
    l_a2=pyplot.plot(np.arange(len(ep_True))[55:72]*0.02,ep_Init[55:72],'k:',label='Initial Model')
    l_a3=pyplot.plot(np.arange(len(ep_True))[55:72]*0.02,ep_FWI1[55:72],'r-',label='Result of Comprehensive FWI Strategies')
    l_a4=pyplot.plot(np.arange(len(ep_True))[55:72]*0.02,ep_FWI2[55:72],'c-.',label='Result of FWI without Multi-scale Strategy')
    l_a5=pyplot.plot(np.arange(len(ep_True))[55:72]*0.02,ep_FWI3[55:72],'y:',label='Result of FWI without Random Excitation Source Strategy')
    l_a6=pyplot.plot(np.arange(len(ep_True))[55:72]*0.02,ep_FWI4[55:72],'m-.',label='Result of FWI without TV Regularization Strategy')
    pyplot.ylim([6,7.5])
    pyplot.savefig('05_FWILine1.png',dpi=1000)