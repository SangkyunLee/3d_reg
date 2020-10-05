#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:53:48 2020

@author: stelioslab
"""

from utils import galvo_corrections, performance
from scipy import signal, ndimage
import numpy as np
import h5py
import os


def est_rasterphase(scan, channel, bskip_field=True, temporal_fill_fraction=0.7129):
    
    """
    scan:  height x width x fields x rep
    temporal_fill_fraction : scan parameter
    """
    
    
    nslices, image_height, image_width, nchannel, nrep = scan.shape
    
    field_ids = np.arange(0,nslices)  
    if bskip_field:
        skip_fields = max(1,int(round(len(field_ids)*0.1)))
        list_fields = field_ids[skip_fields:-2*skip_fields]
    else:
        list_fields = field_ids
    
    taper = np.sqrt(np.outer(signal.tukey(image_height, 0.4),                                     
                             signal.tukey(image_width, 0.4)))
    
    
    raster_phases = []
    for field_id in list_fields:
        slice_=scan[field_id,:, :,channel,:].astype(np.float32, copy=False)
        anscombed = 2 * np.sqrt(slice_ - slice_.min(axis=(0, 1)) + 3 / 8)  # anscombe transform
        template = np.mean(anscombed, axis=-1) * taper
        raster_phases.append(galvo_corrections.compute_raster_phase(template,
                                                                    temporal_fill_fraction))
    raster_phase = np.median(raster_phases)
    raster_std = np.std(raster_phases)
    
    return raster_phase, raster_std

#############################################
    

    


def est_motion_param_stack(scan,raster_phase,fill_fraction,microns_per_pixel,num_processes=10):
    
    nslices, image_height, image_width, nchannel, num_frames = scan.shape
    field_ids = np.arange(0,nslices)
    
    y_shifts = np.zeros([len(field_ids), num_frames])
    x_shifts = np.zeros([len(field_ids), num_frames])
    
    if num_frames>1:
        
        skip_rows = int(round(image_height*0.1))
        skip_cols = int(round(image_width*0.1))
        
        
        # Map: Compute shifts in parallel
        f1 = performance.parallel_motion_stack  # function to map
        
        max_y_shift= max_x_shift = 20 /microns_per_pixel
        
        
        
        
        results = performance.map_fields(f1, scan, field_ids=field_ids,
                                         channel=0,
                                         kwargs={'raster_phase': raster_phase,
                                                 'fill_fraction': fill_fraction,
                                                 'skip_rows': skip_rows,
                                                 'skip_cols': skip_cols,
                                                 'max_y_shift': max_y_shift,
                                                 'max_x_shift': max_x_shift},
                                         num_processes=num_processes,
                                         queue_size=num_processes)
        
        # Reduce: Collect results
        for field_idx, y_shift, x_shift in results:
            y_shifts[field_idx] = y_shift
            x_shifts[field_idx] = x_shift
            
        return (y_shifts,x_shifts)
    
    
    
def est_motion_param_scan(scan,raster_phase,fill_fraction,microns_per_pixel,field_ids=[-1],
                          nframe_template=200,num_processes=10):
    
    nslices, image_height, image_width, nchannel, num_frames = scan.shape
    channel =0
    if field_ids==[-1]:
        field_ids = np.arange(0,nslices)
    
    y_shifts = np.zeros([len(field_ids), num_frames])
    x_shifts = np.zeros([len(field_ids), num_frames])
    
    if num_frames<nframe_template:
        raise IndexError('num_frames is less than 200.')
    else:
        nf_half = int(nframe_template/2)
        
    skip_rows = int(round(image_height*0.1))
    skip_cols = int(round(image_width*0.1))
    
    middle_frame = int(np.floor(num_frames / 2))
    mini_scan = scan[:, skip_rows: -skip_rows, skip_cols: -skip_cols, :,
                     max(middle_frame - nf_half , 0): middle_frame + nf_half ]
    mini_scan = mini_scan.astype(np.float32, copy=False)
    
    
    
    #raster_phase,_ = est_rasterphase(mini_scan, 0,bskip_field=False, temporal_fill_fraction=fill_fraction) 
    
    if raster_phase !=0:        
        for fid in field_ids:
            mini_scan[fid,:,:,channel,:]= galvo_corrections.correct_raster(
                mini_scan[fid,:,:,channel,:], raster_phase, fill_fraction)
            
            
    # Create template
    mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # *
    template = np.mean(mini_scan, axis=-1)[:,:,:,0]
    for fid in field_ids:
        template[fid] = ndimage.gaussian_filter(template[fid], 0.7)  # **
    # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
    # ** Small amount of gaussian smoothing to get rid of high frequency noise  
    
    max_y_shift= max_x_shift = 20 /microns_per_pixel
    
    # Map: Compute shifts in parallel
    f1 = performance.parallel_motion_shifts  # function to map
    
    x_shifts = np.empty((len(field_ids), num_frames))
    y_shifts = np.empty((len(field_ids), num_frames))
    for i, fid in enumerate(field_ids):    
        kwargs= {'raster_phase': raster_phase,'fill_fraction': fill_fraction,
             'template':template[fid]}
        results = performance.map_frames(f1, scan, field_id=fid, channel=0,
                                         y=slice(skip_rows, -skip_rows),
                                         x=slice(skip_cols, -skip_cols),
                                         kwargs=kwargs)
        # Reduce: Collect results
        for frames, chunk_y_shifts, chunk_x_shifts in results:
            y_shifts[i,frames] = chunk_y_shifts
            x_shifts[i,frames] = chunk_x_shifts
  
    
        # detect outliers        
        y_shifts[i], x_shifts[i], outliers = \
        galvo_corrections.fix_outliers(y_shifts[fid], x_shifts[fid], max_y_shift, max_x_shift)
            
    return (y_shifts,x_shifts)    




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
            fn = os.path.join(datapath, fns[i])
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





    
    
def correct_motion(scan, corrected, field_ids, channels, microns_per_pixel,temporal_fill_fraction):    
    """
    correct_motion(scan, corrected, fids, channels, microns_per_pixel,temporal_fill_fraction)
    scan:  fov x height x width x channel x nrep
    corrected = np.empty(scan.shape, dtype=np.float32)
    # channels=0 (default)
    
    output:
        xshift_, yshift_
    """
    

    
    raster_phase,_ =est_rasterphase(scan, channel=0, temporal_fill_fraction=temporal_fill_fraction, bskip_field=False)    
    yshift_, xshift_ = est_motion_param_scan(
        scan,raster_phase,temporal_fill_fraction,microns_per_pixel,
        field_ids = field_ids,num_processes=10)
    # Map: correct shifts in parallel
    f1 = performance.parallel_correct_scan # function to map
    
    for i, fid in enumerate(field_ids):         
        for channel in channels:       
            kwargs= {'raster_phase': raster_phase,'fill_fraction': temporal_fill_fraction,
                 'y_shifts': yshift_[i],'x_shifts': xshift_[i]}
            results = performance.map_frames(f1, scan, field_id=fid, channel=channel,                                    
                                             kwargs=kwargs)
            # Reduce: Collect results
            for frames, chunk in results:
                corrected[i,:,:,channel,frames] = chunk

    return (xshift_, yshift_)
   
        
def save_mc_tseries(datapath, fn, outfn, field_ids, channels, microns_per_pixel,temporal_fill_fraction):    
    """ 
    save_mc_tseries(datapath, fn, outfn, field_ids, channels, microns_per_pixel,temporal_fill_fraction)
    ## 2d t-series ###########
    # datapath ='.'
    # fn ='TSeriesV.mat'
    # microns_per_pixel = 1/0.6 # this value is currently guess and not cruical for motion correction
    # temporal_fill_fraction=0.7129
    # outfn ='MC_Tseries.tif'
    """
    
    fullfn = os.path.join(datapath, fn)    
    fdat = h5py.File(fullfn)
    fn = fn.split('.')[0]
    
    
    scan = fdat[fn]
    
    # add channel dim
    scan = np.expand_dims(scan, axis=len(scan.shape))
    # reorder # fov x height x width x channel x nrep
    scan = np.transpose(scan, axes=(1,3,2,4,0))

    corrected = np.empty(scan.shape, dtype=np.float32)
    mot_par = correct_motion(scan, corrected, field_ids, channels, microns_per_pixel,temporal_fill_fraction)
    
    
        
    from tifffile import imsave
    
    fn, ext = outfn.split('.')
    for fid in field_ids:
        for ch in channels:
            corrected_ = corrected[fid,:,:,ch,:]
            corrected_ = np.transpose(corrected_, axes=(2,0,1))
            
            outfn_ = fn +"_F" +str(fid)+"CH"+str(ch) +"." + ext
            outfullfn = os.path.join(datapath, outfn_)   
            imsave(outfullfn,corrected_)  
            
            corrected_ = np.mean(corrected_, axis =0)
            mean_outfn = "MEAN_"+outfn_
            outfullfn = os.path.join(datapath, mean_outfn)   
            imsave(outfullfn,corrected_)
    
    return mot_par
    
    
    
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
    