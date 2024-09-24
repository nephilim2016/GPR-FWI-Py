#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:26:22 2024

@author: nephilim
"""

import numpy as np
from scipy.signal import firwin,lfilter

def design_fir_filter(cutoff, fs, numtaps):
    cutoff = cutoff
    return firwin(numtaps, cutoff, window='hamming', fs=fs)

def apply_filter(data, fs, cutoff):
    numtaps = int(1 * (fs / (cutoff)))
    fir_coeff = design_fir_filter(cutoff, fs, numtaps)
    filtered_data = lfilter(fir_coeff, 1.0, data)
    return filtered_data

def normalize_amplitude(original_data, filtered_data):
    scale_factor = np.max(np.abs(original_data)) / np.max(np.abs(filtered_data))
    return filtered_data * scale_factor