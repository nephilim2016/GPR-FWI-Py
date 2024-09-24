#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:44:55 2021

@author: nephilim
"""

import Forward2D
import AirForward2D
import Calculate_Gradient_NoSave_Pool
import numpy as np
import time
from Optimization import para,options,Optimization
from pathlib import Path
from matplotlib import pyplot,cm
from skimage import transform
import Create_Model
import RandomSource
import MultiScale

def expand_list(lst):
    expanded_list=[] 
    for item in lst:
        if isinstance(item, list):
            n = item[-2]  
            for idx in range(n):
                new_item = item + [idx] 
                expanded_list.append(new_item)
        else:
            expanded_list.append(item)
    return expanded_list

if __name__=='__main__':  
    start_time=time.time() 
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
    Target_freq=[4e8]
    
    para.Modified_Total_Variation_key=True
    para.initWeight=0.5
    
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
        
    print('Forward Done !')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))
    # Save Profile Data
    para.Original_data=np.load('./%sHz_forward_data_file/record.npy'%para.Freq)
    para.Air_True_Profile_Original=AirForward2D.Forward_2D(np.zeros_like(sigma),np.ones_like(epsilon),np.ones_like(mu),para.Freq,para)
    para.Original_data=para.Original_data-np.tile(para.Air_True_Profile_Original,(para.Original_data.shape[1],1)).T

    # Anonymous function for Gradient Calculate
    fh=lambda x:Calculate_Gradient_NoSave_Pool.misfit(sigma.copy(),x,para)  
    
    # FWI Parameters
    FWI_Params=[]
    Init_Params=[]
    # Random source & Change number
    Source_num_segments=3
    Source_num_start=10
    Source_num_end=22
    Source_Segments,Source_base_size=RandomSource.split_data(para.source_site, num_segments=Source_num_segments)
    Iteration=2
    Source_num_select_list=np.unique(np.linspace(Source_num_start,Source_num_end,Iteration).astype('int'))
    Source_num_select_Record=Source_num_start
    Change_source_num_list=[4,5]
    Change_Iteration_list=[5,5]
    for multi_freq in Target_freq:
        for idx in range(Iteration):
            FWI_Params.append([multi_freq, Source_num_select_list[idx], Change_source_num_list[idx], Change_Iteration_list[idx]])
    
    Init_Params=FWI_Params[:-1]
    Init_Params.insert(0,'Init')
    
    FWI_Params=expand_list(FWI_Params)
    Init_Params=expand_list(Init_Params)
    
    # Starting...
    FWI_INFO=[]
    for idx_FWI in np.arange(len(Init_Params)):
        para.target_freq=FWI_Params[idx_FWI][0]
        para.Source_num_select=FWI_Params[idx_FWI][1]
        para.change_source=FWI_Params[idx_FWI][4]
        if para.target_freq==para.Freq:
            para.MultiScale_Key=False
            para.data=para.Original_data
            para.Air_True_Profile=para.Air_True_Profile_Original
        else:
            para.MultiScale_Key=True
            para.data=np.zeros_like(para.Original_data)
            for idx_data_col in np.arange(para.data.shape[1]):
                para.data[:,idx_data_col]=MultiScale.apply_filter(para.Original_data[:,idx_data_col],para.fs,para.target_freq)
            para.Air_True_Profile=MultiScale.apply_filter(para.Air_True_Profile_Original,para.fs,para.target_freq)
        
        
        para.random_source_site=RandomSource.random_selection_from_segments(Source_Segments, num_to_select=para.Source_num_select)
        para.random_source_index=[para.source_site.index(element) for element in para.random_source_site] 
        if Init_Params[idx_FWI]=='Init':
            iep=np.linspace(1,10,para.xl)
            iep=np.tile(iep,(para.zl,1)).T
            iepsilon=np.zeros((para.xl+20,para.zl+20))
            iepsilon[10:-10,10:-10]=iep
            iepsilon[:CPML,:]=iepsilon[CPML,:]
            iepsilon[-CPML:,:]=iepsilon[-CPML-1,:]
            iepsilon[:,:CPML]=iepsilon[:,CPML].reshape((len(iepsilon[:,CPML]),-1))
            iepsilon[:,-CPML:]=iepsilon[:,-CPML-1].reshape((len(iepsilon[:,-CPML-1]),-1))
            iepsilon[:10+para.AirLayer,:]=1
        else:
            dir_path='./%sHz_imodel_file_%s_%s'%(Init_Params[idx_FWI][0],Init_Params[idx_FWI][1],Init_Params[idx_FWI][4])
            file_num=int(len(list(Path(dir_path).iterdir())))-1
            data=np.load('./%sHz_imodel_file_%s_%s/%s_imodel.npy'%(Init_Params[idx_FWI][0],Init_Params[idx_FWI][1],Init_Params[idx_FWI][4],file_num))
            iepsilon=data.reshape((para.xl+20,-1))
                
        # # Test Gradient
        # f,g=Calculate_Gradient_NoSave_Pool.misfit(sigma.copy(),iepsilon.copy(),para)
        # pyplot.figure()
        # pyplot.imshow(g.reshape((120,-1)))
        
        # Options Params
        options.maxiter=FWI_Params[idx_FWI][3]
        Optimization_=Optimization(fh,iepsilon.copy())
        imodel,info=Optimization_.optimization()
        FWI_INFO.append(info)

        pyplot.figure()
        pyplot.imshow(imodel.reshape((para.xl+20,-1)),cmap=cm.jet)
        pyplot.colorbar()
         
    # Plot Error Data
    pyplot.figure()
    data_=[]
    for info in FWI_INFO:
        for info_ in info:
            data_.append(info_[3])
    pyplot.plot(data_)
    pyplot.yscale('log')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))
    np.save('Loss.npy',data_)
    
    with open('history.txt','w') as fid:
        for dd in FWI_INFO:
            fid.write(str(dd))
            fid.write('\n')
            
    # with open('history.txt', 'r') as fid:
    #     read_list = [eval(line.strip()) for line in fid if line.strip()]