#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:46:22 2020

@author: slee
"""
from tifffile import imread
from utils.Alignment import Alignment3D
from utils.reg_util import save_stackmean, save_scanmean
import os

field_umperpixel = [1/0.8,1/0.8]
stack_umperpixel = [1, 1/0.8, 1/0.8]
temporal_fill_fraction=0.7129

datapath ='./data/'
stackfns =['V0004.mat','V0005.mat','V0006.mat','V0007.mat']
fieldfn ='TSeriesV.mat'
field_outfn ='Tseries_mean.tif'
stack_outfn ='V_stack.tif'

# stack: motion correction save mean image
save_stackmean(datapath, stackfns, stack_outfn, stack_umperpixel[1],temporal_fill_fraction)


# Tseries motion correction and save mean image
save_scanmean(datapath, fieldfn, field_outfn, field_umperpixel[1],temporal_fill_fraction)

# since scan has 3 slices, ImageJ confused to 3 channels and could not open
# therefore, I save each slice separately as below:
# from tifffile import imsave
# out_fn1 = field_fullfn[:-4]+'1.tif'
# imsave(out_fn1,field)



field_fullfn = os.path.join(datapath, field_outfn)
stack_fullfn = os.path.join(datapath, stack_outfn)

####### first slide of Tseries
out_fn = os.path.join(datapath,'Mapping_0')
field = imread(field_fullfn)[0]
stack = imread(stack_fullfn)
stack_z = stack.shape[0]/2
field_z = 50 # this is my guess from visual inspection

Alignment3D(stack, field, out_fn, stack_umperpixel, field_umperpixel, stack_z, field_z)



out_fn = os.path.join(datapath,'Mapping_1')
field = imread(field_fullfn)[1]
stack = imread(stack_fullfn)
stack_z = stack.shape[0]/2
field_z = 150 # this is my guess from visual inspection

Alignment3D(stack, field, out_fn, stack_umperpixel, field_umperpixel, stack_z, field_z)


import pickle
import matplotlib.pyplot as plt
paramfn = os.path.join(datapath,'Mapping_1.pickle')
with open(paramfn, 'rb') as pf:
    par = pickle.load(pf)
affine_grid = par['affine_grid']
plt.matshow(affine_grid[:,:,2])
plt.colorbar()

