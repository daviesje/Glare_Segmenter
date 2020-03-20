import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
from PIL import Image

def read_image(imfile):
    reader = Image.open(imfile,mode='r')
    raster = np.array(reader,dtype=np.uint8)
    return raster

#this is a really roundabout way of joining the datasets using existing functions
#I should specify a new one that doesn't split
IM_DIR = './tiles_seg/test_frames/test/'
LB_DIR = './tiles_seg/test_masks/test/'
im_arr = np.zeros((6,256,256,3),dtype=np.uint8)
lb_arr = np.zeros((6,256,256,3),dtype=np.uint8)
imdir = os.listdir(IM_DIR)
lbdir = os.listdir(LB_DIR)

idx = np.random.randint(len(imdir),size=6)

for i,d in enumerate(idx):
    im_arr[i,...] = read_image(IM_DIR+imdir[d])
    lb_arr[i,...] = read_image(LB_DIR+lbdir[d])

model = load_model(f'./models/model.h5')

#classify new image
#predictions = (model.predict(im_arr/255.) > 0.1)*(np.ones(3)[None,None,None,:])
predictions = model.predict(im_arr/255.)*(np.ones(3)[None,None,None,:])
print(predictions.shape)
print(predictions[0,...].max(),predictions[0,...].min())

#np.save(f'./predictions_{model_name}.npy',predictions)

gs = gridspec.GridSpec(3,6)
fig = plt.figure(figsize=(12,6))

for i in range(predictions.shape[0]):
    ax = fig.add_subplot(gs[0,i])
    if i == 0 : ax.set_ylabel('image')
    ax.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    ax.imshow(im_arr[i,...])
    ax = fig.add_subplot(gs[1,i])
    if i == 0 : ax.set_ylabel('mask')
    ax.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    ax.imshow(lb_arr[i,...])
    ax = fig.add_subplot(gs[2,i])
    if i == 0 : ax.set_ylabel('ML predictions')
    ax.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    ax.imshow(predictions[i,...])

plt.show()
#fig.savefig(f'./predictions_{model_name}.png')