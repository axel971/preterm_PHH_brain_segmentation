
import numpy as np
import tensorflow.keras
from model.one_hot_label import multi_class_labels
from model.dataio import write_label_nii, write_nii
from model.patch3d import patch
from model.image_process import normlize_mean_std
from model.augmentation import create_affine_matrix, similarity_transform_volumes

import torchio as tio

class Generator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, batch_size=32, patch_size=[64, 64, 64], labels=[1], stride=[1, 1, 1]):
        'Initialization'
        self.batch_size = batch_size
        self.patch_size = np.asarray(patch_size)

        self.labels = labels
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.loc_patch = patch(np.asarray(X.shape[1:]), patch_size, stride)

        self.indices_images = np.array(range(self.n_subject))
        self.indices_patches = np.array(range(self.loc_patch.n_patch))
        np.random.shuffle(self.indices_images) # At each epoch the input data are suffle
        np.random.shuffle(self.indices_patches)
        
        if np.any(self.loc_patch.pad_width > [0, 0, 0]): # Pad X and Y if necessary
        	self.X = np.zeros(np.append(len(X),(self.loc_patch.size_after_pad)),  dtype=np.float32)
        	self.Y = np.zeros(np.append(len(Y),(self.loc_patch.size_after_pad)),  dtype=np.float32)
        	
        	for n in range(len(X)):
        		self.X[n] = np.pad(X[n], mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
        		self.Y[n] = np.pad(Y[n], mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
        else:
        	self.X = X
        	self.Y = Y
        
        self.step_by_epoch = 400
        self.current_epoch = 0
        
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(5,5,5), translation = (5, 5, 5)):0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.4, tio.RandomMotion(): 0.2}, p = 0.80) )
    	

    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], len(self.labels)), dtype=np.float32)  # channel_last by default
          
        batch_image_indices = np.random.choice(self.indices_images, self.batch_size)
        batch_patch_indices = np.random.choice(self.indices_patches, self.batch_size)
        
        
        for batch_count in range(self.batch_size):
            image_index = batch_image_indices[batch_count]
            patch_index = batch_patch_indices[batch_count]

            label = self.loc_patch.__get_single_patch__without_padding_test__(self.Y[image_index], patch_index)
            image = self.loc_patch.__get_single_patch__without_padding_test__(self.X[image_index], patch_index)
            
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(image, 0)), segmentation = tio.LabelMap(tensor = np.expand_dims(label, 0)))
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)        
            
            X[batch_count, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            Y[batch_count] = multi_class_labels(patch_tio_augmented['segmentation'].numpy()[0], self.labels) #Perform hot encoding
            
        return X, Y
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
        np.random.shuffle(self.indices_images)
        np.random.shuffle(self.indices_patches)
        

class Generator_with_two_modality(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X1, X2, Y, batch_size=32, patch_size=[64, 64, 64], labels=[1], argmentation=True, stride=[1, 1, 1]):
        'Initialization'
        self.batch_size = batch_size
        self.patch_size = np.asarray(patch_size)
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.labels = labels
        self.image_size = np.asarray(X1.shape[1:])
        self.n_subject = X1.shape[0]
        self.loc_patch = patch(np.asarray(X1.shape[1:]), patch_size, stride)
        tmp = np.indices((self.n_subject, self.loc_patch.n_patch))
        self.indices = np.column_stack((np.ndarray.flatten(tmp[0]), np.ndarray.flatten(tmp[1])))
        np.random.shuffle(self.indices)
        self.step_by_epoch = 250
        self.current_epoch = 0
#         self.total_step = int(np.floor(self.n_subject*self.loc_patch.n_patch/self.batch_size))
#         self.n_epoch_all_data = self.total_step//self.step_by_epoch

        self.argmentation = argmentation
        # self.on_epoch_end()

        # print('n_subject: '+str(self.n_subject))
        # self.loc_patch.__info__()

    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch, batch means how many patch is used once
        X1 = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1), dtype=np.float32)
        X2 = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], len(self.labels)), dtype=np.float32)  # channel_last by default
        # Generate data
           
        sub_index = self.indices[np.random.choice(range(len(self.indices)), self.batch_size)]  # to avoid the index out of the self.indices
        for batch_count in range(self.batch_size):
            image_index = sub_index[batch_count, 0]
            patch_index = sub_index[batch_count, 1]

            label_data = self.Y[image_index]
            label = self.loc_patch.__get_single_patch__(label_data, patch_index)

            image_data1 = self.X1[image_index]
            image_data2 = self.X2[image_index]
            
            image1 = self.loc_patch.__get_single_patch__(image_data1, patch_index)
            image2 = self.loc_patch.__get_single_patch__(image_data2, patch_index)

            # Perform augmentation
            affine, rotation = create_affine_matrix([1,1], [-5,5], [-2,2],  self.patch_size)
            image1, transform = similarity_transform_volumes(image1, affine,  self.patch_size, interpolation = 'continuous')
            image2, transform = similarity_transform_volumes(image2, affine,  self.patch_size, interpolation = 'continuous')
            label, transform = similarity_transform_volumes(label, affine,  self.patch_size, interpolation = 'nearest')
            
            Y[batch_count] = multi_class_labels(label, self.labels)
            X1[batch_count, :, :, :, 0] = image1
            X2[batch_count, :, :, :, 0] = image2
			
        return [X1, X2], Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.current_epoch += 1
        np.random.shuffle(self.indices)
        # np.random.shuffle(self.indices)
        # np.random.shuffle(self.patch_indices)
        # self.indexes = np.arange(len(self.list_IDs))

# 
# 
# class PredictGenerator(keras.utils.Sequence):
#     'Generates data for Keras, based on array data X and Y'
# 
#     def __init__(self, image_data, loc_patch, batch_size=32, patch_size=[32, 32, 32], labels=[1], augmentation=False):
#         'Initialization'
#         self.image_data=image_data
#         self.patch_size = np.asarray(patch_size)
#         self.loc_patch = loc_patch
#         self.patch_index=[]
#         self.on_epoch_end()
# 
#     def __getitem__(self):
#         X = np.zeros((self.patch_size[0], self.patch_size[1], self.patch_size[2], 1), dtype=np.float32)
#         m = 0
#         while m<self.loc_patch.n_patch:
#             image = self.loc_patch.__get_single_patch__(self.image_data, m)
#             image = normlize_mean_std(image)
#             X[:,:,:,0] = image
#             m+=1
#             yield X
# 
#     def on_epoch_end(self):
#         return self.patch_index

