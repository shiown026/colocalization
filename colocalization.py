#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:57:35 2023

@author: chiahunglee

This file is designed to read Zeiss .czi file and 
perform colocalization analysis on 2 channels of fluorescences.
The steps will include follows:
    1. Read .czi and print Metadata
    2. Set Threshold for both channel
    3. Measure and print colocalization and R-value

"""
from os import chdir, getcwd

file_dir = '/Users/chiahunglee/Documents/BurkeLab/Program'
chdir(file_dir)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from aicsimageio import AICSImage


filename = '090723109_chamber4_x63.czi' 

# Get an AICSImage object
img = AICSImage(filename)
img.data  # returns 6D STCZYX numpy array
img.dims  # returns string "STCZYX"
img.shape  # returns tuple of dimension sizes in STCZYX order

print('Image = {}'.format(filename))
print('Number of channels = {}'.format(len(img.channel_names)))
print('Channel names = {}'.format(img.channel_names))
print('Image Size = ({} x {}) px^2 '.format(img.dims.X, img.dims.Y))
print('Pixel Size = {} um'.format(round(img.physical_pixel_sizes.X, 3)))
print('FOV = {} * {} um^2'.format(round(img.physical_pixel_sizes.X*img.dims.X, 3), round(img.physical_pixel_sizes.X*img.dims.X, 3)))


## Get chanel info

ch1 = img.get_image_data("YX", C=0, S=0, T=0)
ch2 = img.get_image_data("YX", C=1, S=0, T=0)


## set threshold

Thre1 = 52
Thre2 = 17

ch1_thre = ch1
ch1_thre=ch1_thre.astype(np.float32)
ch1_thre[ch1_thre<Thre1]=np.nan

ch2_thre = ch2
ch2_thre = ch2_thre.astype(np.float32)
ch2_thre[ch2_thre<Thre2]=np.nan

## flatten
ch1_thre=ch1_thre.flatten()
ch2_thre=ch2_thre.flatten()

## valid points for channels
valid_pts_ch1=len(ch1_thre[~np.isnan(ch1_thre)])
valid_pts_ch2=len(ch2_thre[~np.isnan(ch2_thre)])

## colocalized points
ch3 = ch1_thre*ch2_thre
colocalized_pts = len(ch3[~np.isnan(ch3)])
ch3[~np.isnan(ch3)]=1
colocalized_pxs = ch3
del(ch3)

## colocalize explain
colocalize_ch1 = round(100*colocalized_pts/valid_pts_ch1,3)
print('{}% of the signal in channel 1 is colocalized with channel 2 geometrically.'.format(colocalize_ch1))
colocalize_ch2 = round(100*colocalized_pts/valid_pts_ch2,3)
print('{}% of the signal in channel 2 is colocalized with channel 1 geometrically.'.format(colocalize_ch2))


## Weighted colocalization coefficient
intensity_ch1_col = colocalized_pxs*ch1_thre
intensity_ch1_col =sum(intensity_ch1_col[~np.isnan(intensity_ch1_col)] )

intensity_ch1 = sum(ch1_thre[~np.isnan(ch1_thre)])
Weighted_ch1 = round(100*intensity_ch1_col/intensity_ch1,3)
print('Weighted colocalization:')
print('{}% of the signal in channel 1 is colocalized with channel 2 in terms of the expression level.'.format(Weighted_ch1))

intensity_ch2_col = colocalized_pxs*ch2_thre
intensity_ch2_col =sum(intensity_ch2_col[~np.isnan(intensity_ch2_col)] )

intensity_ch2 = sum(ch2_thre[~np.isnan(ch2_thre)])
Weighted_ch2 = round(100*intensity_ch2_col/intensity_ch2,3)
print('Weighted colocalization:')
print('{}% of the signal in channel 2 is colocalized with channel 1 in terms of the expression level.'.format(Weighted_ch2))

## Pearson Correlation Coefficient (PCC)

from scipy import stats
## stats.pearsonr cannot have infs or NaNs
## Zen Blue use unknown value to replace the NaNs.

ch1_pcc = ch1_thre.copy()
if (np.isnan(ch1_thre).any()):
    ## Use 0
    ch1_pcc[np.isnan(ch1_pcc)]=0

    
    ## Use mean of non-Nan
    #ch1_pcc[np.isnan(ch1_pcc)]=np.mean(ch1_pcc[~np.isnan(ch1_pcc)])
    
    ## Use median of non-Nan
    #ch1_pcc[np.isnan(ch1_pcc)]=np.median(ch1_pcc[~np.isnan(ch1_pcc)])
    

ch2_pcc = ch2_thre.copy()
if (np.isnan(ch2_thre).any()):
    ## Use 0
    ch2_pcc[np.isnan(ch2_pcc)]=0

    
    ## Use mean of non-Nan
    #ch2_pcc[np.isnan(ch2_pcc)]=np.mean(ch2_pcc[~np.isnan(ch2_pcc)])
    
    ## Use median of non-Nan
    #ch2_pcc[np.isnan(ch2_pcc)]=np.median(ch2_pcc[~np.isnan(ch2_pcc)])


res = stats.pearsonr(ch1_pcc,ch2_pcc)
rvalue = round(res[0],4)
print("Pearson's R-value = {} ".format(rvalue))



## set frequency fo pcc scatter plot
data={'ch1_pcc': ch1_pcc,'ch2_pcc': ch2_pcc, 'unique': ch1_pcc}
df=pd.DataFrame(data=data,index=None)
for i in range(df.shape[0]):
    df['unique'][i]=str(df.ch1_pcc[i])+'-'+str(df.ch2_pcc[i])

df['unique'].nunique()
sns.scatterplot(data=data, x='ch1_pcc',y='ch2_pcc', hue='unique')


# Manders Overlap Coefficient (MOC)


