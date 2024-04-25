import os
import numpy as np
from model import unet3d
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from generator_array import Generator
from dataio import printgpu, import_data_filename, write_nii
from image_process import load_image_correct_oritation, crop_pad3D, resize
#from viewer import view3plane
from one_hot_label import redefine_label
import metrics
import metrics2
import time
import pickle
from prediction_my import evaluate, test
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
import pandas
import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
from utils import get_nii_data, get_nii_affine, save_image

def main():
    n_subject = 17  # len(subject_index)
    subject_list = np.arange(n_subject)
    image_size = [256, 156, 256]
    patch_size = [64, 64, 64]
    labels = [0, 1, 2, 3, 4, 5, 6]  # redefine labels

    ###########################################################################
    # read data
    file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name.xlsx"
    data = pandas.read_excel(file_dir, index=False)
    data_length = data.shape
    subject_names = np.array(data['subject_name'])
    img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing'
    delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
    model_weight_path = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/evaluation_hyper_parameters/weights/'
 	
    np.random.shuffle(subject_list)
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]),
                 dtype=np.float32)
    Y = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]),
                 dtype=np.float32)
    affines = []
                 
    for n in range(n_subject):
        X[n] = get_nii_data(img_dir + '/' + subject_names[n] + '_preproc.nii.gz' )
        Y[n] = get_nii_data(delineation_dir + '/' + subject_names[n] + '_delineations.nii.gz' )
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '_preproc.nii.gz'))
        
    affines = np.array(affines)
    
    n_voxel = image_size[0] * image_size[1] * image_size[2]

    class_percent = []
    
    for iClass in range(len(labels)):
    	class_percent_by_subject  = 0
    	
    	for iSubject in range(len(Y)):
    		class_percent_by_subject = class_percent_by_subject + (np.count_nonzero(Y[iSubject] == iClass) / n_voxel)
    	class_percent.append(class_percent_by_subject/len(Y))
    
    print(class_percent)

if __name__ == '__main__':
    main()
