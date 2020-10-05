#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:32:21 2020

@author: root
"""


from utils.reg_util import save_mc_tseries
import pickle
import os

field_umperpixel = [1/0.8,1/0.8]
temporal_fill_fraction=0.7129

datapath ='./data/'

fieldfn ='TSeriesV.mat'
field_outfn ='MC_TseriesV.tif'
field_ids = [0]
channels = [0]
# Tseries motion correction and save motion corrected images and mean image
mot_par = save_mc_tseries(datapath, fieldfn, field_outfn, field_ids, channels, field_umperpixel[1],temporal_fill_fraction)

motpar_fn = "motionpar" + fieldfn.split('.')[0] +".pickle"
motpar_fn = os.path.join(datapath, motpar_fn)    
pickle.dump( mot_par, open( motpar_fn, "wb" ) )




########################################
from sklearn.decomposition import NMF

