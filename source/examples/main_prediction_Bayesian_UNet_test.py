import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
import os
import tensorflow as tf
from model.model import bayesian_Unet3d
from tensorflow.keras import optimizers
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator
import numpy as np
from model.prediction_my import BayesianPredict
from model.image_process import crop_edge_pair, load_image_correct_oritation
from model.dataio import import_data_filename, write_nii
from model.one_hot_label import redefine_label
import metrics.metrics as metrics
import time
import pickle
import gc
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

from utils import get_nii_data, get_nii_affine, save_image
from imgaug import augmenters as iaa

def main():
    n_subject = 1
    subject_list = np.arange(n_subject)
    image_size = [256, 128, 256]
    patch_size = [64, 64, 64]
    labels = [0, 1, 2, 3, 4, 5, 6]  # redefine labels
    # np.random.shuffle(subject_list)
    
    # Get path data
    img_dir = '/home/axel/dev/neonatal_brain_segmentation/source/examples/test'
    subject_names='onesie_1z105_02_recon_050'
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    affines = []
    
    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names + '_preproc.nii.gz' )
        affines.append(get_nii_affine(img_dir + '/' + subject_names + '_preproc.nii.gz'))
    toc = time.clock()

    print('..data size:' + str(X.shape), flush=True)
    print('..loading data finished ' + str(time.asctime(time.localtime())), flush=True)
    
    
    
    model = tf.keras.models.load_model('/home/axel/dev/neonatal_brain_segmentation/source/examples/test/model_training_whole_cohort.h5', custom_objects = {'dice_multi': metrics.dice_multi})    	
    	    	
    predictor = BayesianPredict(model, image_size, patch_size, labels)
    	
    for i in range(len(X)):
    	prediction, MCsamples = predictor.__run__(X[i])
    	save_image(prediction, affines[i],  '/home/axel/dev/neonatal_brain_segmentation/source/examples/test/' + subject_names + '_predicted_segmentation.nii')
     	  
    del model
    	
    K.clear_session()
    gc.collect()

    
if __name__ == '__main__':
    main()
