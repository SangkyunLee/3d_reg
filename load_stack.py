#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:22:33 2020

@author: stelioslab
"""

import h5py
import numpy as np
import os
from utils import performance

## 3d stack ###########
datapath ='.'
ext = '.mat'
fns =['V0004','V0005','V0006','V0007']
nproc =10
microns_per_pixel = 1/0.8 # this value is currently guess and not cruical for motion correction
temporal_fill_fraction=0.7129
outfilename ='V_stack.tif'


########################
fn = os.path.join(datapath, fns[0]+ext)
fdat = h5py.File(fn)

nslice, width, height =fdat[fns[0]].shape[1:4]
slices = list(range(0,nslice,nproc))

if nslice<nproc:
    nproc = nslice
corrected = np.empty((nslice, height, width), dtype = np.float32)
for ix, ss in enumerate(slices):
    for i in range(len(fns)):
        fn = os.path.join(datapath, fns[i]+ext)
        fdat = h5py.File(fn)          
    
        if ix == len(slices)-1:
            d = fdat[fns[i]][:, ss:,:,:]        
            depth_ids = list(range(ss,nslice))                
        else:
            d = fdat[fns[i]][:, ss:ss+nproc,:,:]  
            depth_ids = list(range(ss,ss+nproc))
        
        if i==0:
            stack =d            
        else:
            stack = np.concatenate((stack, d), axis=0)
            

        

    stack = np.expand_dims(stack, axis=len(stack.shape))
    # reorder # fov x height x width x channel x nrep
    stack = np.transpose(stack, axes=(1,3,2,4,0))
    
    from utils.reg_util import est_rasterphase, est_motion_param_stack
    raster_phase,_ =est_rasterphase(stack, 0, temporal_fill_fraction)    
    yshift_, xshift_ = est_motion_param_stack(stack,raster_phase,temporal_fill_fraction,microns_per_pixel,
                                        num_processes=nproc)

    
        
    hf = performance.parallel_correct_stack
    results = performance.map_fields(hf, stack, field_ids=list(range(stack.shape[0])), channel=0,
                                 kwargs={'raster_phase': raster_phase,
                                         'fill_fraction': temporal_fill_fraction,
                                         'y_shifts': yshift_,
                                         'x_shifts': xshift_},
                                 num_processes=nproc, queue_size=nproc)
    
    
    for field_idx, corrected_field in results:
        corrected[depth_ids[field_idx]] = corrected_field
        
from tifffile import imsave
imsave(outfilename,corrected)        


# imsave('Tseries_stack2.tif',corrected[2])        


    
        