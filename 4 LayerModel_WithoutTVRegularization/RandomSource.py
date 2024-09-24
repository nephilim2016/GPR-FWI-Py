#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 06:31:58 2024

@author: nephilim
"""

import numpy as np
import random

def split_data(data, num_segments):
    arr_len = len(data)
    base_size = int(np.ceil(arr_len / num_segments))
    remainder = arr_len % base_size

    segments = []
    start = 0
    for i in range(num_segments):
        size = base_size if i < num_segments else remainder
        segments.append(data[start:start + size])
        start += size

    return segments,base_size

def random_selection_from_segments(segments, num_to_select=1):
    last_segment = segments[-1]
    selected_elements = []
    
    remaining_to_select = len(last_segment) if len(last_segment)<num_to_select else num_to_select

    for segment in segments[:-1]:

        selected_element = random.sample(list(segment), num_to_select)
        selected_element.sort(key=lambda x: list(segment).index(x))
        
        selected_elements.extend(selected_element)
    
    segment=segments[-1]
    selected_element = random.sample(list(segment), remaining_to_select)
    selected_element.sort(key=lambda x: list(segment).index(x))
    selected_elements.extend(selected_element)
    
    return selected_elements