#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:59:51 2018

@author: nephilim
"""
import numpy as np

#Ricker wavelet   
def ricker(t,f):
    w=-1*(2*np.pi**2*(f*t-1)**2-1)*np.exp(-np.pi**2*(f*t-1)**2)
    return w