#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:45:57 2021

@author: nephilim
"""

from numba import jit
import numpy as np

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
def reverse_time_loop(xl,zl,dx,dz,dt,sigma,epsilon,mu,CPML_Params,k_max,ref_pos,rhs_data):
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
    ca = CPML_Params.ca_r
    cb = CPML_Params.cb
            
    for tt in range(k_max):
        Ey[ref_pos[0],ref_pos[1]]-=rhs_data[k_max-tt-1]
        Hz,Hx=update_H(xl,zl,dx,dz,dt,sigma,epsilon,mu,npml,a_x_half,a_z_half,b_x_half,b_z_half,k_x_half,k_z_half,Hz,Hx,Ey,memory_dEy_dx,memory_dEy_dz)
        Ey=update_E(xl,zl,dx,dz,dt,ca,cb,npml,a_x,a_z,b_x,b_z,k_x,k_z,Hz,Hx,Ey,memory_dHz_dx,memory_dHx_dz)
        # pyplot.imshow(Ey,vmin=-50,vmax=50)
        # pyplot.pause(0.01)
        yield Ey.tolist(),
