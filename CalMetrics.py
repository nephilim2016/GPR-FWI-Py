#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:04:21 2025

@author: nephilim
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def plot_dual_metrics(groups, metric1_values, metric2_values, 
                     metric1_name="Metric1", metric2_name="Metric2",
                     group_names=None,
                     y1_range=None,  
                     y2_range=None): 


    
    color1 = 'blue'
    color2 = 'green'
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    bar_width = 0.35
    group_positions = np.arange(groups)
    
    bars1 = ax1.bar(group_positions - bar_width/2, metric1_values, bar_width, 
                    label=metric1_name, color=color1, alpha=0.3)
    bars2 = ax2.bar(group_positions + bar_width/2, metric2_values, bar_width,
                    label=metric2_name, color=color2, alpha=0.3)
    
    line1 = ax1.plot(group_positions - bar_width/2, metric1_values, 'o-', 
                     color=color1, label=f'{metric1_name}', linewidth=2)
    line2 = ax2.plot(group_positions + bar_width/2, metric2_values, 's-', 
                     color=color2, label=f'{metric2_name}', linewidth=2)
    
    def add_value_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
    
    add_value_labels(bars1, ax1)
    add_value_labels(bars2, ax2)
    
    ax1.set_xticks(group_positions)
    ax1.set_xticklabels(group_names)
    
    ax1.set_xlabel('Strategy')
    
    ax1.set_ylabel(metric1_name, color=color1)
    ax2.set_ylabel(metric2_name, color=color2)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    if y1_range is not None:
        ax1.set_ylim(y1_range)
    if y2_range is not None:
        ax2.set_ylim(y2_range)
    # plt.tight_layout()
    plt.grid('on')
    return fig, (ax1, ax2)

def compare_matrices(matrix1, matrix2):
    m1_flat = matrix1.flatten()
    m2_flat = matrix2.flatten()

    # Pearson Correlation
    pearson_corr, _ = pearsonr(m1_flat, m2_flat)

    # SSIM
    ssim = calculate_ssim(matrix1, matrix2)
    
    return {
        'pearson_correlation': pearson_corr,
        'ssim': ssim
    }

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    window = np.zeros((window_size, window_size))
    sigma = 1.5
    gauss = np.exp(-((np.arange(window_size) - window_size//2)**2) / (2*sigma**2))
    gauss = gauss/gauss.sum()
    window = np.outer(gauss, gauss)
    
    K1 = 0.01
    K2 = 0.03
    L = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    
    mu1 = np.zeros_like(img1, dtype=np.float64)
    mu2 = np.zeros_like(img2, dtype=np.float64)
    
    for i in range(window_size//2, img1.shape[0]-window_size//2):
        for j in range(window_size//2, img1.shape[1]-window_size//2):
            mu1[i,j] = np.sum(img1[i-window_size//2:i+window_size//2+1, 
                                 j-window_size//2:j+window_size//2+1] * window)
            mu2[i,j] = np.sum(img2[i-window_size//2:i+window_size//2+1, 
                                 j-window_size//2:j+window_size//2+1] * window)
    
    sigma1_sq = np.zeros_like(img1, dtype=np.float64)
    sigma2_sq = np.zeros_like(img2, dtype=np.float64)
    sigma12 = np.zeros_like(img1, dtype=np.float64)
    
    for i in range(window_size//2, img1.shape[0]-window_size//2):
        for j in range(window_size//2, img1.shape[1]-window_size//2):
            sigma1_sq[i,j] = np.sum(((img1[i-window_size//2:i+window_size//2+1, 
                                          j-window_size//2:j+window_size//2+1] - mu1[i,j])**2) * window)
            sigma2_sq[i,j] = np.sum(((img2[i-window_size//2:i+window_size//2+1, 
                                          j-window_size//2:j+window_size//2+1] - mu2[i,j])**2) * window)
            sigma12[i,j] = np.sum(((img1[i-window_size//2:i+window_size//2+1, 
                                        j-window_size//2:j+window_size//2+1] - mu1[i,j]) *
                                  (img2[i-window_size//2:i+window_size//2+1, 
                                        j-window_size//2:j+window_size//2+1] - mu2[i,j])) * window)
    
    ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return np.mean(ssim_map)
    else:
        return ssim_map

if __name__ == "__main__":
    epsilon=np.load('./1 LayerModelFWI/LayerModel.npy')
    
    data=np.load('./1 LayerModelFWI/400000000.0Hz_imodel_file_22_4/4_imodel.npy')
    data=data.reshape((120,220))
    FWIdata=data[10:-10,10:-10]
    
    data=np.load('./2 LayerModel_WithoutMultiply/400000000.0Hz_imodel_file_22_4/4_imodel.npy')
    data=data.reshape((120,220))
    FWIdata_Mul=data[10:-10,10:-10]
    
    data=np.load('./3 LayerModel_WithoutRandomSource/400000000.0Hz_imodel_file_100_0/19_imodel.npy')
    data=data.reshape((120,220))
    FWIdata_Ran=data[10:-10,10:-10]
    
    data=np.load('./4 LayerModel_WithoutTVRegularization/400000000.0Hz_imodel_file_22_4/4_imodel.npy')
    data=data.reshape((120,220))
    FWIdata_TV=data[10:-10,10:-10]
    
    value0=compare_matrices(epsilon[35:85,135:185], FWIdata[35:85,135:185])
    value1=compare_matrices(epsilon[35:85,135:185], FWIdata_Mul[35:85,135:185])
    value2=compare_matrices(epsilon[35:85,135:185], FWIdata_Ran[35:85,135:185])
    value3=compare_matrices(epsilon[35:85,135:185], FWIdata_TV[35:85,135:185])    
    
    groups=4
    metric1_values = [value0['pearson_correlation'], value1['pearson_correlation'], value2['pearson_correlation'], value3['pearson_correlation']]
    metric2_values = [value0['ssim'], value1['ssim'], value2['ssim'], value3['ssim']] 
    group_names = ['S1', 'S2', 'S3', 'S4']
    
    fig, axes = plot_dual_metrics(
        groups=groups,
        metric1_values=metric1_values,
        metric2_values=metric2_values,
        metric1_name='Pearson Correlation Coefficient',
        metric2_name='SSIM',
        group_names=group_names,
        y1_range=(0.95,1),  
        y2_range=(0.75,1))
    
    plt.savefig('Add_Metric.png',dpi=1000)