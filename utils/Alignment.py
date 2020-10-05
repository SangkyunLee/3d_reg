#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:03:15 2020

@author: slee
"""


import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from skimage import feature
from skimage import filters




def Alignment3D(stack, field, out_fn, stack_umperpixel, field_umperpixel,
               stack_z, field_z, opts=None):
    
    """
    field_umperpixel = [1/0.8,1/0.8]
    stack_umperpixel = [1, 1/0.8, 1/0.8]
    field_fn = '../data/Tseries_mean.tif'
    stack_fn = '../data/V_stack.tif'
    out_fn = '../data/Mapping_0'
    field = imread(field_fn)[0]
    stack = imread(stack_fn)
    
    stack_z = stack.shape[0]/2
    field_z = 50 # this is my guess from visual inspection
    
    saved files:
        params : paramfn = out_fn+'.pickle'
        params['affine_linear'] : rigid body model affine linear
        params['affine_translation'] : rigid body model affine translation
        params['nonrigid_linear'] : nonrigid_linear
        params['nonrigid_translation'] : nonrigid_translation
        params['affine_grid'] : grid estimated from rigid + non-rigid affine transformation
        
        params['deformations'] : warping parameters        
        params['warping_field'] : warping field from estimated from deformation
        params['corr'] : correlation between predicted field and original field
        params['pred_field'] : field sampled from pre-processed stack with predicted grid
        params['pred_field0'] : field sampled from non-processed (only with resmapling) stack with predicted grid
        
    outfullfn = out_fn+'-processed.tif'
        save params['pred_field'] 
    outfullfn = out_fn+'.tif'
        save params['pred_field0'] 
    outfullfn = out_fn+'field.tif'
        save field image sampled in 1um resolution
        
    """
    
    field, field0 = pre_process(field,field_umperpixel,[15,15])
    stack, stack0 = pre_process(stack, stack_umperpixel,(3,25,25))
    
    assert field.shape>(60,60), "field is too small"
    assert stack.shape>(20,60,60), "stack is too small"
    field = field[15:-15, 15:-15]    
    stack = stack[5:-5, 15:-15, 15:-15]
    stack0 = stack0[5:-5, 15:-15, 15:-15]
    field0 = field0[15:-15, 15:-15]    
    # Set params
    cparam={}
    cparam['rigid_zrange'] = 80  # microns to search above and below estimated z for rigid registration
    cparam['lr_linear'] = 0.001  # learning rate / step size for the linear part of the affine matrix
    cparam['lr_translation'] = 1  # learning rate / step size for the translation vector
    cparam['affine_iters'] = 200  # number of optimization iterations to learn the affine parameters
    cparam['random_seed'] = 1234  # seed for torch random number generator (used to initialize deformations)
    cparam['landmark_gap'] = 100  # spacing for the landmarks
    cparam['rbf_radius'] = 150  # critical radius for the gaussian rbf
    cparam['lr_deformations'] = 0.1  # learning rate / step size for deformation values
    cparam['wd_deformations'] = 1e-4  # weight decay for deformations; controls their size
    cparam['smoothness_factor'] = 0.01  # factor to keep the deformation field smooth
    cparam['nonrigid_iters'] = 200  # number of optimization iterations for the nonrigid parameters
    
    if opts:
        for key, val in opts.items():
            cparam[key] = val


    
    # Run registration without rotation
    px_z = field_z - stack_z + stack.shape[0] / 2 - 0.5
    mini_stack = stack[max(0, int(round(px_z - cparam['rigid_zrange']))): int(round(
        px_z + cparam['rigid_zrange']))]
    corrs = np.stack([feature.match_template(s, field, pad_input=True) for s in
                      mini_stack])
   
    smooth_corrs = filters.gaussian(corrs,sigma=0.7)
    
    # Get results
    min_z = max(0, int(round(px_z - cparam['rigid_zrange'])))
    min_y = int(round(0.05 * stack.shape[1]))
    min_x = int(round(0.05 * stack.shape[2]))
    mini_corrs = smooth_corrs[:, min_y:-min_y, min_x:-min_x]
    rig_z, rig_y, rig_x = np.unravel_index(np.argmax(mini_corrs), mini_corrs.shape)
    
    # Rewrite coordinates with respect to original z
    rig_z = (min_z + rig_z + 0.5) - stack.shape[0] / 2
    rig_y = (min_y + rig_y + 0.5) - stack.shape[1] / 2
    rig_x = (min_x + rig_x + 0.5) - stack.shape[2] / 2
    
    del px_z, mini_stack, corrs, smooth_corrs, min_z, min_y, min_x, mini_corrs
        
    grid = create_grid(field.shape)
    
    
    # Create torch tensors
    stack_ = torch.as_tensor(stack, dtype=torch.float32)
    field_ = torch.as_tensor(field, dtype=torch.float32)
    grid_ = torch.as_tensor(grid, dtype=torch.float32)
    
    
    
    # Define parameters and optimizer
    linear = torch.nn.Parameter(torch.eye(3)[:, :2])  # first two columns of rotation matrix
    translation = torch.nn.Parameter(torch.tensor([rig_x, rig_y, rig_z]))  # translation vector
    affine_optimizer = optim.Adam([{'params': linear, 'lr': cparam['lr_linear']},
                                   {'params': translation, 'lr': cparam['lr_translation']}])
    
    
    
    # Optimize
    for i in range(cparam['affine_iters']):
        # Zero gradients
        affine_optimizer.zero_grad()
    
        # Compute gradients
        pred_grid = affine_product(grid_, linear, translation)  # w x h x 3
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
    torch.manual_seed(cparam['random_seed'])  # we use random initialization below
    
    # Create landmarks (and their corresponding deformations)
    first_y = int(round((field.shape[0] % cparam['landmark_gap']) / 2))
    first_x = int(round((field.shape[1] % cparam['landmark_gap']) / 2))
    landmarks = grid_[first_x::cparam['landmark_gap'],
                  first_y::cparam['landmark_gap']].contiguous().view(-1, 2)  # num_landmarks x 2
    
    # Compute rbf scores between landmarks and grid coordinates and between landmarks
    grid_distances = torch.norm(grid_.unsqueeze(-2) - landmarks, dim=-1)
    grid_scores = torch.exp(-(grid_distances * (1 / cparam['rbf_radius'])) ** 2)  # w x h x num_landmarks
    landmark_distances = torch.norm(landmarks.unsqueeze(-2) - landmarks, dim=-1)
    landmark_scores = torch.exp(-(landmark_distances * (1 / 200)) ** 2)  # num_landmarks x num_landmarks
    
    # Define parameters and optimizer
    deformations = torch.nn.Parameter(torch.randn((landmarks.shape[0], 3)) / 10)  # N(0, 0.1)
    nonrigid_optimizer = optim.Adam([deformations], lr=cparam['lr_deformations'],
                                weight_decay=cparam['wd_deformations'])
    
    # Optimize
    for i in range(cparam['nonrigid_iters']):
        # Zero gradients
        affine_optimizer.zero_grad()  # we reuse affine_optimizer so the affine matrix changes slowly
        nonrigid_optimizer.zero_grad()
        
        # Compute grid with radial basis
        affine_grid = affine_product(grid_, linear, translation)
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
        loss = corr_loss + cparam['smoothness_factor'] * reg_term
        print('Corr/loss at iteration {}: {:5.4f}/{:5.4f}'.format(i, -corr_loss,
                                                                  loss))
        loss.backward()
        
        # Update
        affine_optimizer.step()
        nonrigid_optimizer.step()
        
    # Save final results
    nonrigid_linear = linear.detach().clone()
    nonrigid_translation = translation.detach().clone()
    # nonrigid_landmarks = landmarks.clone()
    # nonrigid_deformations = deformations.detach().clone()
    
    pred_field = sample_grid(stack_, pred_grid)
    stack0_ = torch.as_tensor(stack0, dtype=torch.float32) # original stack after resample at 1um but without preprocess
    pred_field0 = sample_grid(stack0_, pred_grid)
    
    
    params ={}
    params['affine_linear'] =affine_linear.detach().numpy()  # rigid body model
    params['affine_translation'] = affine_translation.detach().numpy() 
    params['nonrigid_linear'] = nonrigid_linear.detach().numpy()
    params['nonrigid_translation'] = nonrigid_translation.detach().numpy()
    
    params['deformations'] = deformations.detach().numpy()
    params['affine_grid'] = affine_grid.detach().numpy()
    params['warping_field'] = warping_field.detach().numpy()
    params['corr'] = -1*corr_loss.detach().numpy()
    params['pred_field'] = pred_field.detach().numpy()
    params['pred_field0'] = pred_field0.detach().numpy()   
    
    import pickle
    paramfn = out_fn+'.pickle'
    with open(paramfn, 'ab') as pf:
        pickle.dump(params,pf)
    

    from tifffile import imsave  
    outfullfn = out_fn+'-processed.tif'
    imsave(outfullfn,params['pred_field']) # preprocessed file
    
    outfullfn = out_fn+'.tif'
    imsave(outfullfn,params['pred_field0'])
    outfullfn = out_fn+'field.tif'
    imsave(outfullfn,field0)
    
def pre_process(field, umperpixel, lcnpar):
    
    import  enhancement  
    um_sizes=tuple(np.array(field.shape)*np.array(umperpixel))
    desired_res =1
    
    resized = resize(field, um_sizes,desired_res)
    
     # Enhance
    lcned = enhancement.lcn(resized, lcnpar)
    
    # Sharpen
    sharpened = enhancement.sharpen_2pimage(lcned, 1)
    return sharpened, resized
      



def create_grid(um_sizes, desired_res=1):
    """ Create a grid corresponding to the sample position of each pixel/voxel in a FOV of
     um_sizes at resolution desired_res. The center of the FOV is (0, 0, 0).

    In our convention, samples are taken in the center of each pixel/voxel, i.e., a volume
    centered at zero of size 4 will have samples at -1.5, -0.5, 0.5 and 1.5; thus edges
    are NOT at -2 and 2 which is the assumption in some libraries.

    :param tuple um_sizes: Size in microns of the FOV, .e.g., (d1, d2, d3) for a stack.
    :param float or tuple desired_res: Desired resolution (um/px) for the grid.

    :return: A (d1 x d2 x ... x dn x n) array of coordinates. For a stack, the points at
    each grid position are (x, y, z) points; (x, y) for fields. Remember that in our stack
    coordinate system the first axis represents z, the second, y and the third, x so, e.g.,
    p[10, 20, 30, 0] represents the value in x at grid position 10, 20, 30.
    """
    # Make sure desired_res is a tuple with the same size as um_sizes
    if np.isscalar(desired_res):
        desired_res = (desired_res,) * len(um_sizes)

    # Create grid
    out_sizes = [int(round(um_s / res)) for um_s, res in zip(um_sizes, desired_res)]
    um_grids = [np.linspace(-(s - 1) * res / 2, (s - 1) * res / 2, s, dtype=np.float32)
                for s, res in zip(out_sizes, desired_res)] # *
    full_grid = np.stack(np.meshgrid(*um_grids, indexing='ij')[::-1], axis=-1)
    # * this preserves the desired resolution by slightly changing the size of the FOV to
    # out_sizes rather than um_sizes / desired_res.

    return full_grid


def resize(original, um_sizes, desired_res):
    """ Resize array originally of um_sizes size to have desired_res resolution.

    We preserve the center of original and resized arrays exactly in the middle. We also
    make sure resolution is exactly the desired resolution. Given these two constraints,
    we cannot hold FOV of original and resized arrays to be exactly the same.

    :param np.array original: Array to resize.
    :param tuple um_sizes: Size in microns of the array (one per axis).
    :param int or tuple desired_res: Desired resolution (um/px) for the output array.

    :return: Output array (np.float32) resampled to the desired resolution. Size in pixels
        is round(um_sizes / desired_res).
    """
    import torch.nn.functional as F

    # Create grid to sample in microns
    grid = create_grid(um_sizes, desired_res) # d x h x w x 3

    # Re-express as a torch grid [-1, 1]
    um_per_px = np.array([um / px for um, px in zip(um_sizes, original.shape)])
    torch_ones = np.array(um_sizes) / 2 - um_per_px / 2  # sample position of last pixel in original
    grid = grid / torch_ones[::-1].astype(np.float32)

    # Resample
    input_tensor = torch.from_numpy(original.reshape(1, 1, *original.shape).astype(
        np.float32))
    grid_tensor = torch.from_numpy(grid.reshape(1, *grid.shape))
    resized_tensor = F.grid_sample(input_tensor, grid_tensor, padding_mode='border')
    resized = resized_tensor.numpy().squeeze()

    return resized


def affine_product(X, A, b):
    """ Special case of affine transformation that receives coordinates X in 2-d (x, y)
    and affine matrix A and translation vector b in 3-d (x, y, z). Y = AX + b

    :param torch.Tensor X: A matrix of 2-d coordinates (d1 x d2 x 2).
    :param torch.Tensor A: The first two columns of the affine matrix (3 x 2).
    :param torch.Tensor b: A 3-d translation vector.

    :return: A (d1 x d2 x 3) torch.Tensor corresponding to the transformed coordinates.
    """
    return torch.einsum('ij,klj->kli', (A, X)) + b


def sample_grid(volume, grid):
    """ Volume is a d x h x w arrray, grid is a d1 x d2 x 3 (x, y, z) coordinates
    and output is a d1 x d2 array"""
    norm_factor = torch.as_tensor([s / 2 - 0.5 for s in volume.shape[::-1]])
    norm_grid = grid / norm_factor  # between -1 and 1
    resampled = F.grid_sample(volume.view(1, 1, *volume.shape),
                              norm_grid.view(1, 1, *norm_grid.shape),
                              padding_mode='zeros')
    return resampled.squeeze()    

