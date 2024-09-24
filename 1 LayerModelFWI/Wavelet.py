#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:59:51 2018

@author: nephilim
"""
import numpy as np
def ricker(t,f):
    w=-1*(2*np.pi**2*(f*t-1)**2-1)*np.exp(-np.pi**2*(f*t-1)**2)
    return w
    
def blackmanharris(t,f):
    a=[0.3532222,-0.488,0.145,-0.010222222]
    T=1.14/f
    f_=np.zeros(len(t))
    for n in range(len(a)):
        f_=f_+a[n]*np.cos(2*n*np.pi*t/T)
    f_[t>=T]=0
    f_tmp=f_[1:]
    f_tmp=np.append(f_tmp,0)
    srcpulse=f_tmp-f_
    srcpulse/=np.max(np.abs(srcpulse))
    return srcpulse   
    
def gauss(t,f):
    a=[0.3532222,-0.488,0.145,-0.010222222]
    T=1.14/f
    f_=np.zeros(len(t))
    for n in range(len(a)):
        f_=f_+a[n]*np.cos(2*n*np.pi*t/T)
    f_[t>=T]=0

    return f_    
