import os
import numpy as np
from model import unet3d
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import multi_gpu_model
from generator_array import Generator
from dataio import printgpu, import_data_filename, write_nii
from image_process import load_image_correct_oritation, crop_pad3D, resize
#from viewer import view3plane
from one_hot_label import redefine_label
import metrics
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

def main(argv):
    n_subject = 17  # len(subject_index)
    subject_list = np.arange(n_subject)
    image_size = [256, 156, 256]
    patch_size = [64, 64, 64]
    labels = [0, 1, 2]  # redefine labels

    ###########################################################################
    # read data
    file_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name.xlsx"
    data = pandas.read_excel(file_dir, index=False)
    data_length = data.shape
    subject_names = np.array(data['subject_name'])
    img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/preprocessing'
    delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/ventricle'
    output_img_dir = argv[0]
    model_weight_path = argv[1]

 	
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
    
    #############################################################
    # Get the model
    single_model = unet3d(patch_size+[1], len(labels))
    model = multi_gpu_model(single_model, gpus=2)
    model.load_weights(model_weight_path)
    
    #########################################################
    # Perform prediction
    ID = 8  # 10-fold
    prediction = test(X[ID:, ...], model, image_size, patch_size, labels)
    
    for iPrediction in range(len(prediction)):
    	save_image(prediction[iPrediction], affines[iPrediction], output_img_dir + str(subject_name[ID + iPrediction]) + '.nii')

if __name__ == '__main__':
    main(sys.argv[1:])
