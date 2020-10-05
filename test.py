#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:08:27 2019

@author: stelioslab
"""




import h5py
import numpy as np
import matplotlib.pyplot as plt

fns = ['stack_50_v1', 'stack50_v2']
ext ='.mat'
stack = [None]*len(fns)
for i in range(len(fns)):
    fdat = h5py.File(fns[i]+ext)
    stack[i] = np.array(fdat[fns[i]])
    stack[i] = stack[i].transpose()

if len(stack)>1:
    stack = np.concatenate(stack, axis=3)    
else:
    stack = stack[0]

height,width,nslices,nvol = stack.shape

# re-order matrix dimension
# fov x height x width x channel x nrep
stack = np.expand_dims(stack,axis =len(stack.shape))
stack = np.moveaxis(stack,np.arange(0,len(stack.shape)),[1,2,0,4,3])




fdat = h5py.File('T.mat','r')
Tseries = np.array(fdat['Tseries'])
Tseries = Tseries.transpose()

# re-order matrix dimension
# fov x height x width x channel x nrep
Tseries = np.expand_dims(Tseries,axis =len(Tseries.shape))
Tseries = np.moveaxis(Tseries,np.arange(0,len(Tseries.shape)),[1,2,0,4,3])
    



from pipeline.utils import galvo_corrections, stitching, performance, enhancement
from scipy import signal

def est_rasterphase(scan, channel, temporal_fill_fraction=0.7129):
    
    """
    scan:  height x width x fields x rep
    temporal_fill_fraction : scan parameter
    """
    
    
    nslices, image_height, image_width, nchannel, nrep = scan.shape
    
    field_ids = np.arange(0,nslices)    
    skip_fields = max(1,int(round(len(field_ids)*0.1)))
    
    taper = np.sqrt(np.outer(signal.tukey(image_height, 0.4),                                     
                             signal.tukey(image_width, 0.4)))
    
    
    raster_phases = []
    for field_id in field_ids[skip_fields:-2*skip_fields]:
        slice_=scan[field_id,:, :,channel,:].astype(np.float32, copy=False)
        anscombed = 2 * np.sqrt(slice_ - slice_.min(axis=(0, 1)) + 3 / 8)  # anscombe transform
        template = np.mean(anscombed, axis=-1) * taper
        raster_phases.append(galvo_corrections.compute_raster_phase(template,
                                                                    temporal_fill_fraction))
    raster_phase = np.median(raster_phases)
    raster_std = np.std(raster_phases)
    
    return raster_phase, raster_std

#############################################
    

    
##############################################    
    
from pipeline.utils import galvo_corrections, performance

def est_motion_param(scan,raster_phase,fill_fraction,microns_per_pixel):
    
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
                                                 'max_x_shift': max_x_shift})
        
        # Reduce: Collect results
        for field_idx, y_shift, x_shift in results:
            y_shifts[field_idx] = y_shift
            x_shifts[field_idx] = x_shift
            
        return (y_shifts,x_shifts)
      
        
    
    
    
    
    
    
# apply motion correction to image volume
fill_fraction = 0.7129
microns_per_pixel = 1/0.8
raster_phase,_ =est_rasterphase(stack, 0, temporal_fill_fraction=0.7129)
yshifts, xshifts = est_motion_param(stack,raster_phase,fill_fraction,microns_per_pixel)


field_ids = list(range(nslices))        
hf = performance.parallel_correct_stack
results = performance.map_fields(hf, stack, field_ids=field_ids, channel=0,
                                 kwargs={'raster_phase': raster_phase,
                                         'fill_fraction': fill_fraction,
                                         'y_shifts': yshifts,
                                         'x_shifts': xshifts})

corrected_roi = np.empty((stack.shape[:3]), dtype = np.float32)
for field_idx, corrected_field in results:
    corrected_roi[field_idx]=corrected_field
from tifffile import imsave
imsave('corrected_stack.tif',corrected_roi)







################ motion correction for 2d scan ########

        
fill_fraction = 0.7129
microns_per_pixel = 1/0.6

raster_phase,_ =est_rasterphase(Tseries, 0, temporal_fill_fraction=0.7129)


yshifts, xshifts = est_motion_param(Tseries,raster_phase,fill_fraction,microns_per_pixel)


nslices = Tseries.shape[0]
field_ids = list(range(nslices))        
hf1 = performance.parallel_correct_stack
Tresult = performance.map_fieldx-s(hf1, Tseries, field_ids=field_ids, channel=0,
                                 kwargs={'raster_phase': raster_phase,
                                         'fill_fraction': fill_fraction,
                                         'y_shifts': yshifts,
                                         'x_shifts': xshifts})


corrected_roi = np.empty((Tseries.shape[:3]), dtype = np.float32)
for field_idx, corrected_field in Tresult:
    corrected_roi[field_idx]=corrected_field
from tifffile import imsave
imsave('corrected_Tseries.tif',corrected_roi)    

from tifffile import imread
corrected_roi = imread('corrected_stack.tif')
        
from pipeline.utils import galvo_corrections, stitching, performance, enhancement    


################### pre-process field and stack
from tifffile import imread
import numpy as np
from pipeline.utils import galvo_corrections, performance, enhancement   
field0 = imread('./data/Tseries_mean.tif')
stack0 = imread('./data/V_stack.tif')



def pre_process(field, umperpixel, lcnpar):
    um_sizes=tuple(np.array(field.shape)*np.array(umperpixel))
    desired_res =1
    
    
    import pipeline.utils.registration as registration
    resized = registration.resize(field, um_sizes,desired_res)
    
     # Enhance
    lcned = enhancement.lcn(resized, lcnpar)
    
    # Sharpen
    sharpened = enhancement.sharpen_2pimage(lcned, 1)
    return sharpened


field0 = pre_process(field0[0],[1/0.8,1/0.8],[15,15])
stack0 = pre_process(stack0, [1, 1/0.8, 1/0.8],(3,25,25))


field = field0[15:-15, 15:-15]
stack = stack0[5:-5, 15:-15, 15:-15]

#stack_z = np.arange(stack.shape[0]*1.0)
stack_z = stack.shape[0]/2
field_z = 50 # this is my guess from visual inspection


# Set params
rigid_zrange = 80  # microns to search above and below estimated z for rigid registration
lr_linear = 0.001  # learning rate / step size for the linear part of the affine matrix
lr_translation = 1  # learning rate / step size for the translation vector
affine_iters = 200  # number of optimization iterations to learn the affine parameters
random_seed = 1234  # seed for torch random number generator (used to initialize deformations)
landmark_gap = 100  # spacing for the landmarks
rbf_radius = 150  # critical radius for the gaussian rbf
lr_deformations = 0.1  # learning rate / step size for deformation values
wd_deformations = 1e-4  # weight decay for deformations; controls their size
smoothness_factor = 0.01  # factor to keep the deformation field smooth
nonrigid_iters = 200  # number of optimization iterations for the nonrigid parameters


from skimage import feature
from skimage import filters
from scipy import ndimage

# Run registration with no rotations
px_z = field_z - stack_z + stack.shape[0] / 2 - 0.5
mini_stack = stack[max(0, int(round(px_z - rigid_zrange))): int(round(
    px_z + rigid_zrange))]
corrs = np.stack([feature.match_template(s, field, pad_input=True) for s in
                  mini_stack])
#smooth_corrs = ndimage.gaussian_filter(corrs, 0.7)

smooth_corrs = filters.gaussian(corrs,sigma=0.7)



# Get results
min_z = max(0, int(round(px_z - rigid_zrange)))
min_y = int(round(0.05 * stack.shape[1]))
min_x = int(round(0.05 * stack.shape[2]))
mini_corrs = smooth_corrs[:, min_y:-min_y, min_x:-min_x]
rig_z, rig_y, rig_x = np.unravel_index(np.argmax(mini_corrs), mini_corrs.shape)

# Rewrite coordinates with respect to original z
rig_z = (min_z + rig_z + 0.5) - stack.shape[0] / 2
rig_y = (min_y + rig_y + 0.5) - stack.shape[1] / 2
rig_x = (min_x + rig_x + 0.5) - stack.shape[2] / 2

del px_z, mini_stack, corrs, smooth_corrs, min_z, min_y, min_x, mini_corrs




# Affine Registration
from utils import registration
import torch
from torch import optim
import torch.nn.functional as F

def sample_grid(volume, grid):
    """ Volume is a d x h x w arrray, grid is a d1 x d2 x 3 (x, y, z) coordinates
    and output is a d1 x d2 array"""
    norm_factor = torch.as_tensor([s / 2 - 0.5 for s in volume.shape[::-1]])
    norm_grid = grid / norm_factor  # between -1 and 1
    resampled = F.grid_sample(volume.view(1, 1, *volume.shape),
                              norm_grid.view(1, 1, *norm_grid.shape),
                              padding_mode='zeros')
    return resampled.squeeze()

grid = registration.create_grid(field.shape)


# Create torch tensors
stack_ = torch.as_tensor(stack, dtype=torch.float32)
field_ = torch.as_tensor(field, dtype=torch.float32)
grid_ = torch.as_tensor(grid, dtype=torch.float32)



# Define parameters and optimizer
linear = torch.nn.Parameter(torch.eye(3)[:, :2])  # first two columns of rotation matrix
translation = torch.nn.Parameter(torch.tensor([rig_x, rig_y, rig_z]))  # translation vector
affine_optimizer = optim.Adam([{'params': linear, 'lr': lr_linear},
                               {'params': translation, 'lr': lr_translation}])






from pipeline.utils import registration
# Optimize
for i in range(affine_iters):
    # Zero gradients
    affine_optimizer.zero_grad()

    # Compute gradients
    pred_grid = registration.affine_product(grid_, linear, translation)  # w x h x 3
    pred_field = sample_grid(stack_, pred_grid)
    corr_loss = -(pred_field * field_).sum() / (torch.norm(pred_field) *
                                                torch.norm(field_))
    print('Corr at iteration {}: {:5.4f}'.format(i, -corr_loss))
    corr_loss.backward()

    # Update
    affine_optimizer.step()

# Save them (originals will be modified during non-rigid registration)
affine_linear = linear.detach().clone()
affine_translation = translation.detach().clone()


# NON-RIGID REGISTRATION
# Inspired by the the Demon's Algorithm (Thirion, 1998)
torch.manual_seed(random_seed)  # we use random initialization below

# Create landmarks (and their corresponding deformations)
first_y = int(round((field.shape[0] % landmark_gap) / 2))
first_x = int(round((field.shape[1] % landmark_gap) / 2))
landmarks = grid_[first_x::landmark_gap,
              first_y::landmark_gap].contiguous().view(-1, 2)  # num_landmarks x 2

# Compute rbf scores between landmarks and grid coordinates and between landmarks
grid_distances = torch.norm(grid_.unsqueeze(-2) - landmarks, dim=-1)
grid_scores = torch.exp(-(grid_distances * (1 / rbf_radius)) ** 2)  # w x h x num_landmarks
landmark_distances = torch.norm(landmarks.unsqueeze(-2) - landmarks, dim=-1)
landmark_scores = torch.exp(-(landmark_distances * (1 / 200)) ** 2)  # num_landmarks x num_landmarks

# Define parameters and optimizer
deformations = torch.nn.Parameter(torch.randn((landmarks.shape[0], 3)) / 10)  # N(0, 0.1)
nonrigid_optimizer = optim.Adam([deformations], lr=lr_deformations,
                            weight_decay=wd_deformations)

# Optimize
for i in range(nonrigid_iters):
    # Zero gradients
    affine_optimizer.zero_grad()  # we reuse affine_optimizer so the affine matrix changes slowly
    nonrigid_optimizer.zero_grad()
    
    # Compute grid with radial basis
    affine_grid = registration.affine_product(grid_, linear, translation)
    warping_field = torch.einsum('whl,lt->wht', (grid_scores, deformations))
    pred_grid = affine_grid + warping_field
    pred_field = sample_grid(stack_, pred_grid)
    
    # Compute loss
    corr_loss = -(pred_field * field_).sum() / (torch.norm(pred_field) *
                                                torch.norm(field_))
    
    # Compute cosine similarity between landmarks (and weight em by distance)
    norm_deformations = deformations / torch.norm(deformations, dim=-1,
                                                  keepdim=True)
    cosine_similarity = torch.mm(norm_deformations, norm_deformations.t())
    reg_term = -((cosine_similarity * landmark_scores).sum() /
                 landmark_scores.sum())
    
    # Compute gradients
    loss = corr_loss + smoothness_factor * reg_term
    print('Corr/loss at iteration {}: {:5.4f}/{:5.4f}'.format(i, -corr_loss,
                                                              loss))
    loss.backward()
    
    # Update
    affine_optimizer.step()
    nonrigid_optimizer.step()

import os
from tifffile import imsave
datapath='./data/'
outfn='V_stack_field0'
outfullfn = os.path.join(datapath, outfn+'.tif')  
outfield = pred_field.detach().numpy() 
imsave(outfullfn,outfield)

# linear = linear.detach().numpy()
# translation = translation.detach().numpy() 
# deformation = deformations.detach().numpy()
# affine_grid = affine_grid.detach().numpy()
# warping_field = warping_field.detach().numpy()
# corr = -1*corr_loss.detach().numpy()

params ={}
params['linear'] =linear.detach().numpy()
params['translation'] = translation.detach().numpy() 
params['deformation'] = deformations.detach().numpy()
params['affine_grid'] = affine_grid.detach().numpy()
params['warping_field'] = warping_field.detach().numpy()
params['corr'] = -1*corr_loss.detach().numpy()

import pickle
paramfn = os.path.join(datapath, outfn+'.pickle')  
with open(paramfn, 'ab') as pf:
    pickle.dump(params,pf)

with open(paramfn, 'rb') as pf:
    par = pickle.load(pf)


######################################################


# Create landmarks (and their corresponding deformations)
first_y = int(round((field.shape[0] % landmark_gap) / 2))
first_x = int(round((field.shape[1] % landmark_gap) / 2))
landmarks = grid_[first_x::landmark_gap, first_y::landmark_gap].contiguous().view(
    -1, 2)  # num_landmarks x 2

# Compute rbf scores between landmarks and grid coordinates and between landmarks
rbf_radius = 150
grid_distances = torch.norm(grid_.unsqueeze(-2) - landmarks, dim=-1)
grid_scores = torch.exp(-(grid_distances * (1 / rbf_radius)) ** 2)  # w x h x num_landmarks
landmark_distances = torch.norm(landmarks.unsqueeze(-2) - landmarks, dim=-1)
landmark_scores = torch.exp(-(landmark_distances * (1 / 200)) ** 2)  # num_landmarks x num_landmarks

# Define parameters and optimizer
wd_deformations = 1e-4  # weight decay for deformations; controls their size
deformations = torch.nn.Parameter(torch.randn((landmarks.shape[0], 3)) / 10)  # N(0, 0.1)



# desired_res = (desired_res,) * len(um_sizes)
# out_sizes = [int(round(um_s / res)) for um_s, res in zip(um_sizes, desired_res)]
# um_grids = [np.linspace(-(s - 1) * res / 2, (s - 1) * res / 2, s, dtype=np.float32)
#                 for s, res in zip(out_sizes, desired_res)] # *


# grid0 = np.stack(np.meshgrid(*um_grids, indexing='ij')[::-1], axis=-1)


# import torch
# import torch.nn.functional as F

# um_per_px = np.array([um / px for um, px in zip(um_sizes, original.shape)])
# torch_ones = np.array(um_sizes) / 2 - um_per_px / 2  # sample position of last pixel in original

# grid = grid0 / torch_ones[::-1].astype(np.float32)


# input_tensor = torch.from_numpy(original.reshape(1, 1, *original.shape).astype(np.float32))
# grid_tensor = torch.from_numpy(grid.reshape(1, *grid.shape))
# # F.grid_sample need grid input in the coordinate order of x,y,z
# resized_tensor = F.grid_sample(input_tensor, grid_tensor, padding_mode='border')
# resized = resized_tensor.numpy().squeeze()
# plt.matshow(resized[0,:,:])




