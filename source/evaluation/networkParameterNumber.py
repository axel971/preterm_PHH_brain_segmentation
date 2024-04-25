import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
import os
import tensorflow as tf
from model.model import bayesian_Unet3d
from tensorflow.keras import optimizers
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import backend as K
from model.ConcreteDropout.spatialConcreteDropout import Spatial3DConcreteDropout
import numpy as np

import metrics.metrics as metrics
import time
import pickle
import gc

# from viewer import view3plane
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


def main():
	
	model = tf.keras.models.load_model('/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet_spatial_concrete_dropout/exp1/models/model_1.h5', custom_objects = {'dice_multi': metrics.dice_multi, 'Spatial3DConcreteDropout': Spatial3DConcreteDropout})
	model.summary()   	
    	    	
    
if __name__ == '__main__':
    main()
