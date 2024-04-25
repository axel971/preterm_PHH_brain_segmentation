import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
import os
from model.ConcreteDropout.spatialConcreteDropout import Spatial3DConcreteDropout
import tensorflow as tf
tf.executing_eagerly()
import numpy as np
import time
import pickle
import gc
import metrics.metrics as metrics
from sklearn.model_selection import KFold
# from viewer import view3plane
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
import pandas
import math
from utils import get_nii_data, get_nii_affine, save_image
from imgaug import augmenters as iaa

def main():

    
    nFold = 4

    for iFold in range(nFold):

    	
    	model = tf.keras.models.load_model('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp1/models/model_' + str(iFold) + '.h5', custom_objects = {'dice_multi': metrics.dice_multi, 'Spatial3DConcreteDropout': Spatial3DConcreteDropout})
    	    	
    	print("----- Fold " + str(iFold) + "-------")
    
#     	for layer in model.get_layer('model').layers: 
    	for layer in model.layers:    		
    		if (hasattr(layer, 'p')):
    			print(1 / (1 + math.exp(-layer.get_weights()[0])))
    			
    
if __name__ == '__main__':
    main()