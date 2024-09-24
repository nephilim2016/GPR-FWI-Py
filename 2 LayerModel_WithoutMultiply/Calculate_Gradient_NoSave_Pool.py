#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:20:43 2021

@author: nephilim
"""

from multiprocessing import Pool
import numpy as np
import time
import Add_CPML
import Wavelet
import Time_loop
import Reverse_time_loop
import Modified_TV
import MultiScale

def calculate_gradient(sigma,epsilon,mu,index,CPML_Params,para):
    #Get Forward Params
    k_max=para.k_max
    ep0 = 8.841941282883074e-12
    t=np.arange(k_max)*para.dt
    f=Wavelet.ricker(t,para.Freq)
    if para.MultiScale_Key:
        f=MultiScale.apply_filter(f, para.fs, para.target_freq)
    #True Model Profile Data
    data=para.data[:,index]
    #Get Forward Data ----> <Generator>
    Forward_data=Time_loop.time_loop(para.xl,para.zl,para.dx,para.dz,para.dt,\
                                     sigma.copy(),epsilon.copy(),mu.copy(),CPML_Params,f,k_max,\
                                     para.source_site[index],para.receiver_site[index])
    #Get Generator Data
    For_data=[]
    idata=np.zeros(para.k_max)
    for idx in range(para.k_max):
        tmp=Forward_data.__next__()
        For_data.append(np.array(tmp[0]))
        idata[idx]=tmp[1]
    idata-=para.Air_True_Profile
    #Get Residual Data
    rhs_data=idata-data
    #Get Reversion Data ----> <Generator>
    Reverse_data=Reverse_time_loop.reverse_time_loop(para.xl,para.zl,para.dx,para.dz,\
                                                     para.dt,sigma.copy(),epsilon.copy(),\
                                                     mu.copy(),CPML_Params,k_max,\
                                                     para.receiver_site[index],rhs_data)
    #Get Generator Data
    RT_data=[]
    for i in range(para.k_max):
        tmp=Reverse_data.__next__()
        RT_data.append(np.array(tmp[0]))
    RT_data.reverse()
    
    time_sum_eps=np.zeros((para.xl+2*CPML_Params.npml,para.zl+2*CPML_Params.npml))

    for k in range(1,k_max-1):
        u1=For_data[k+1]
        u0=For_data[k-1]
        p1=RT_data[k]
        time_sum_eps+=p1*(u1-u0)/para.dt/2

    g_eps=ep0*time_sum_eps
    
    g_eps[:10+para.AirLayer,:]=0

    return rhs_data.flatten(),g_eps.flatten()    

#Calculate Modified-Total-Variation
def calculate_mtv_model(epsilon):
    u_epsilon=Modified_TV.denoising_2D_TV(epsilon)
    return u_epsilon

def calculate_mtv_penalty_data(epsilon,u_epsilon):
    epsilon_rhs=epsilon-u_epsilon
    f_epsilon=0.5*np.linalg.norm(epsilon_rhs.flatten(),2)**2
    g_epsilon=2*(epsilon-u_epsilon)
    return f_epsilon,g_epsilon.flatten()

def misfit(sigma,epsilon,para): 
    mu=np.ones((para.xl+20,para.zl+20))
    start_time=time.time()  
    CPML_Params=Add_CPML.Add_CPML(para.xl,para.zl,sigma.copy(),epsilon.copy(),mu.copy(),para.dx,para.dz,para.dt)
    g_eps=0.0
    rhs=[]
    pool=Pool(processes=128)
    res_l=[]
    
    for index,value in zip(para.random_source_index,para.random_source_site):
        res=pool.apply_async(calculate_gradient,args=(sigma.copy(),epsilon.copy(),mu.copy(),index,CPML_Params,para))
        res_l.append(res)
    pool.close()
    pool.join()

    for res in res_l:
        result=res.get()
        rhs.append(result[0])
        g_eps+=result[1]
        del result
    rhs=np.array(rhs)        
    f=0.5*np.linalg.norm(rhs.flatten(),2)**2
    
    #Get Modified Toltal Variation
    if para.Modified_Total_Variation_key:
        mtv_time=time.time()
        u_epsilon=calculate_mtv_model(epsilon.copy())
        f_mtv_penalty_epsilon,g_mtv_penalty_epsilon=calculate_mtv_penalty_data(epsilon,u_epsilon)
        print('mtv function elapsed time is %s seconds !'%str(time.time()-mtv_time))
    
        print('''****fd=%s,g=%s,f_mtv_penalty_epsilon=%s,g_mtv_penalty_epsilon=%s'''\
              %(f,np.linalg.norm(g_eps,2),f_mtv_penalty_epsilon,np.linalg.norm(g_mtv_penalty_epsilon,2)))
            #Update Lambda
        lambda_=(np.linalg.norm(g_eps.flatten(),2))/(np.linalg.norm(g_mtv_penalty_epsilon.flatten(),2))*para.initWeight
        
        f+=lambda_*f_mtv_penalty_epsilon
        g_eps+=lambda_*g_mtv_penalty_epsilon
        
        
    pool.terminate() 
#    print('**********',lambda_,'**********')
    print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
    return f,g_eps