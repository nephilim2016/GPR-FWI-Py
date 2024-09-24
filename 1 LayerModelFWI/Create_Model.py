#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:58:37 2018

@author: nephilim
"""
from numba import jit
import numpy as np
from skimage import filters

@jit(nopython=True)
def Abnormal_Model(xl,zl):
    epsilon=np.ones((xl,zl))*4
    p=34
    l=14
    w=3
    for i in range(p-l,p+l):
        for j in range(p-w,p+w):
            epsilon[i][j]=1
    for i in range(p-w,p+w):
        for j in range(p-l,p+l):
            epsilon[i][j]=1
    p = zl-p
    for i in range(p-l,p+l):
        for j in range(p-w,p+w):
            epsilon[i][j]=8
    for i in range(p-w,p+w):
        for j in range(p-l,p+l):
            epsilon[i][j]=8
    CPML=10
    epsilon_=np.ones((xl+2*CPML,zl+2*CPML))*4
    epsilon_[10:-10,10:-10]=epsilon
    epsilon_[:CPML,:]=epsilon_[CPML,:]
    epsilon_[-CPML:,:]=epsilon_[-CPML-1,:]
    epsilon_[:,:CPML]=epsilon_[:,CPML].reshape((len(epsilon_[:,CPML]),-1))
    epsilon_[:,-CPML:]=epsilon_[:,-CPML-1].reshape((len(epsilon_[:,-CPML-1]),-1))
    return epsilon

#Create Initial_Overthrust Model
def Initial_Smooth_Model(epsilon_,sig):
    iepsilon=filters.gaussian(epsilon_,sigma=sig)
    return iepsilon