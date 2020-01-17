# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:47:34 2020

@author: jed12
"""
from PIL import Image
import numpy as np

def blockshaped(arr,subh,subw):
    h,w = arr.shape[:2]
    return (arr.reshape(h//subh,subh,-1,subw)
            .swapaxes(1,2)
            .reshape(-1,subh,subw))

def read_image(imfile):
    reader = Image.open(imfile,mode='r')
    raster = np.array(reader)
    return raster

fullimage = read_image('../Glare_Classifier/tiles/masks/Bleaching_glare_polygon_big_04_11.png')
buf = np.zeros((256,256,256,3))
buf[...,0] = blockshaped(fullimage[...,0],256,256)
buf[...,1] = blockshaped(fullimage[...,1],256,256)
buf[...,2] = blockshaped(fullimage[...,2],256,256)

buf = buf.astype('uint8')

for i in range(256):
    im = Image.fromarray(buf[i,...])
    im.save(f'./tiles_seg/labels/Glare_04_11_{i:03d}.png')