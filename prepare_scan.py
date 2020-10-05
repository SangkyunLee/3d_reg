#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:38:41 2020

@author: root
"""

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










########################
def save_stackmean(datapath, fns, outfn, microns_per_pixel,temporal_fill_fraction, nproc=10):    
    """
    def save_stackmean(datapath, fns, outfn, microns_per_pixel,temporal_fill_fraction, nproc=10)
    ## save 3d stack ###########
    datapath ='./data/'
    
    fns =['V0004.mat','V0005.mat','V0006.mat','V0007.mat']
    nproc =10
    microns_per_pixel = 1/0.8 # this value is currently guess and not cruical for motion correction
    temporal_fill_fraction=0.7129
    outfn ='V_stack.tif'
    """
    fn = os.path.join(datapath, fns[0])
    fdat = h5py.File(fn)
    
    
    fldname = fns[0].split('.')[0]
    nslice, width, height =fdat[fldname].shape[1:4]
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
    outfullfn = os.path.join(datapath, outfn)   
    imsave(outfullfn,corrected)




def save_scanmean(datapath, fn, outfn, microns_per_pixel,temporal_fill_fraction):    
    """ 
    get_scanmean(datapath, fn, outfn, microns_per_pixel,temporal_fill_fraction)
    ## 2d t-series ###########
    # datapath ='.'
    # fn ='TSeriesV.mat'
    # microns_per_pixel = 1/0.6 # this value is currently guess and not cruical for motion correction
    # temporal_fill_fraction=0.7129
    # outfn ='Tseries_mean.tif'
    """
    
    fullfn = os.path.join(datapath, fn)    
    fdat = h5py.File(fullfn)
    fn = fn.split('.')[0]
    
    
    scan = fdat[fn]
    
    # add channel dim
    scan = np.expand_dims(scan, axis=len(scan.shape))
    # reorder # fov x height x width x channel x nrep
    scan = np.transpose(scan, axes=(1,3,2,4,0))
    channel=0
    corrected = np.empty(scan.shape, dtype=np.float32)
    
    
    
    from utils.reg_util import est_rasterphase, est_motion_param_scan
    raster_phase,_ =est_rasterphase(scan, channel=channel, temporal_fill_fraction=temporal_fill_fraction, bskip_field=False)    
    yshift_, xshift_ = est_motion_param_scan(scan,raster_phase,temporal_fill_fraction,microns_per_pixel,
                                    num_processes=10)
    
    # Map: correct shifts in parallel
    f1 = performance.parallel_correct_scan # function to map
    for fid in range(scan.shape[0]):    
        kwargs= {'raster_phase': raster_phase,'fill_fraction': temporal_fill_fraction,
             'y_shifts': yshift_[fid],'x_shifts': xshift_[fid]}
        results = performance.map_frames(f1, scan, field_id=fid, channel=channel,                                    
                                         kwargs=kwargs)
    
    # Reduce: Collect results
    for frames, chunk in results:
        corrected[fid,:,:,channel,frames] = chunk
    
    corrected = np.mean(np.mean(corrected, axis =4),axis=3)
        
    from tifffile import imsave
    outfullfn = os.path.join(datapath, outfn)   
    imsave(outfullfn,corrected)  
      


   


    
        