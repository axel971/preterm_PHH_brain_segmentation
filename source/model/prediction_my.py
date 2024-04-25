import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')
import tensorflow as tf
import numpy as np
from model.patch3d import patch
from model.image_process import normlize_mean_std, crop_pad3D, crop3D_hotEncoding
import metrics.metrics as metrics
from model.dataio import write_nii
from model.one_hot_label import restore_labels
import time
# 
# class predict(object):
#     'run the model on test data'
# 
#     def __init__(self, model, image_size, patch_size=(32, 32, 32), labels=[1]):
#         'Initialization'
#         self.patch_size = patch_size
#         self.labels = labels
#         self.stride = [14, 14, 14]
#         self.image_size = np.asarray(image_size)
#         self.model = model
#         self.loc_patch = patch(self.image_size, patch_size, self.stride)
#         self.batch_size = 2
#         
#         
#     def __run__(self, X):
#     	'test on one image each time'
#     	
#     	# Pad the input image X if necessary
#     	if np.any(self.loc_patch.pad_width > [0, 0, 0]):
#     		X_pad = np.pad(X, mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
#     	else:
#     		X_pad = X
#     	
#     	
#     	Y0 = np.zeros(np.append((self.loc_patch.size_after_pad),len(self.labels)),  dtype=np.float32) #Initialization: padded uncertainty map array
#     	X0 = np.zeros([self.loc_patch.n_patch]+self.patch_size+[1]) #Initialization: patches used as batch 
#     	
#     	 
#     	for index in range(self.loc_patch.n_patch):
#     		X0[index, :,:,:,0] = self.loc_patch.__get_single_patch__without_padding_test__(X_pad, index)# Why not write do this process in one sentences (why the use of newaxis is required) ?
#     	X0 = tf.convert_to_tensor(X0)
#     	prediction = self.model.predict(X0,batch_size = 2)
# #     	predict_tf_gen = tf.data.Dataset.from_tensor_slices(X0)
# #     	predict_tf_gen = predict_tf_gen.batch(self.batch_size)		
# #     	prediction = self.model.predict(predict_tf_gen)
# 
# 
#     	for index in range(self.loc_patch.n_patch):
#     		Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction[index]), index)
#     		
#     	Y0 = crop3D_hotEncoding(Y0, self.image_size, len(self.labels))	 
#     	
#     	return restore_labels(Y0, self.labels) #return Y0 without hot encoding
    	

class predict(object):
    'run the model on test data'

    def __init__(self, model, image_size, patch_size=(32, 32, 32), labels=[1]):
        'Initialization'
        self.patch_size = patch_size
        self.labels = labels
        self.stride = [14, 14, 14]
        self.image_size = np.asarray(image_size)
        self.model = model
        self.loc_patch = patch(self.image_size, patch_size, self.stride)
        self.batch_size = 6
        
        
    def __run__(self, X):
    	'test on one image each time'
    	
    	# Pad the input image X if necessary
    	if np.any(self.loc_patch.pad_width > [0, 0, 0]):
    		X_pad = np.pad(X, mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
    	else:
    		X_pad = X
    	   	
#     	output_size = np.append((self.loc_patch.size_after_pad),len(self.labels))
    	Y0 = np.zeros(np.append((self.loc_patch.size_after_pad),len(self.labels)),  dtype=np.float32) #Initialization: padded uncertainty map array
    	X0 = np.zeros([self.batch_size]+self.patch_size+[1]) #Initialization: patches used as batch 
    	
    	const_array = np.asarray(range(self.batch_size)) #Initialize: array with elements are iteratively equal to 0 until batch_size-1
    	 
    	for index in range(np.ceil(self.loc_patch.n_patch/self.batch_size).astype(int)):
    		batch_of_patch_index =  const_array + self.batch_size*index
    		batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
    		
    		# get one batch_size
    		for n, selected_patch in enumerate(batch_of_patch_index):
    			X0[n, :,:,:,0] = self.loc_patch.__get_single_patch__without_padding_test__(X_pad, selected_patch)
    			
    		prediction = self.model(X0, training = False)

    		for n, selected_patch in enumerate(batch_of_patch_index):
    			Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction[n]), selected_patch)
    		
    	Y0 = crop3D_hotEncoding(Y0, self.image_size, len(self.labels))	 
    	
    	return restore_labels(Y0, self.labels) #return Y0 without hot encoding
  
class BayesianPredict(object):
    'run the model on test data'

    def __init__(self, model, image_size, patch_size=(32, 32, 32), labels=[1], T = 6):
        'Initialization'
        self.patch_size = patch_size
        self.labels = labels
        self.stride = [14, 14, 14]
        self.image_size = np.asarray(image_size)
        self.model = model
        self.loc_patch = patch(self.image_size, patch_size, self.stride)
        self.T = T
        self.batch_size = 6

    def __run__(self, X):
    	'test on one image each time'
    
		# Pad the input image X if necessary
    	if np.any(self.loc_patch.pad_width > [0, 0, 0]):
    		X_pad = np.pad(X, mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
    	else:
    		X_pad = X
    	   	
    	   	
    	output_size = np.append((self.loc_patch.size_after_pad), len(self.labels))
    	output_size = np.append(self.T, output_size)
    	
    	Y0 = np.zeros(output_size,  dtype=np.float32)
    	X0 = np.zeros([self.batch_size]+self.patch_size+[1])
    	
    	const_array = np.asarray(range(self.batch_size)) #Initialize: array with elements are iteratively equal to 0 until batch_size-1
    	
    			  			  	
    	for iPass in range(self.T):
    	
    		for index in range(np.ceil(self.loc_patch.n_patch/self.batch_size).astype(int)):
    			batch_of_patch_index = const_array + self.batch_size*index
    			batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
    		
    			# get one batch_size
    			for n, selected_patch in enumerate(batch_of_patch_index):
    				X0[n, :,:,:,0] = self.loc_patch.__get_single_patch__without_padding_test__(X_pad, selected_patch)
    			
    			# predict the segmentation patches for the current batch
    			prediction = self.model(X0, training = False)
   		
    			# put the label back
    			for n, selected_patch in enumerate(batch_of_patch_index):
    				Y0[iPass] = self.loc_patch.__put_single_patch__(Y0[iPass], np.squeeze(prediction[n]), selected_patch) #Put single patch perform patch-wise addition
    		
    	
    	# Compute the mean of the predicted probability maps
    	YProbaMap = np.sum(Y0, axis = 0)
#     	Y0 = Y0 / np.sum(Y0, axis = -1, keepdims = True)
#     	YProbaMap = np.sum(Y0, axis = 0)
    	YProbaMap = crop3D_hotEncoding(YProbaMap, self.image_size, len(self.labels))
    	Y = restore_labels(YProbaMap, self.labels)
    	
    	return Y, Y0
    	
    	
class ensembleLearningPredict(object):
    'run the model on test data'

    def __init__(self, model1, model2, model3, model4, image_size, patch_size=(32, 32, 32), labels=[1]):
        'Initialization'
        self.patch_size = patch_size
        self.labels = labels
        self.stride = [14, 14, 14]
        self.image_size = np.asarray(image_size)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.loc_patch = patch(self.image_size, patch_size, self.stride)
        self.batch_size = 6

    def __run__(self, X):
    	'test on one image each time'
    
		# Pad the input image X if necessary
    	if np.any(self.loc_patch.pad_width > [0, 0, 0]):
    		X_pad = np.pad(X, mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
    	else:
    		X_pad = X
    	
    	Y0 = np.zeros(np.append((self.loc_patch.size_after_pad),len(self.labels)),  dtype=np.float32) #Initialization: padded uncertainty map array
    	X0 = np.zeros([self.batch_size]+self.patch_size+[1]) #Initialization: patches used as batch 
    	
    	const_array = np.asarray(range(self.batch_size)) #Initialize: array with elements are iteratively equal to 0 until batch_size-1
    	
    			  			  	

    	for index in range(np.ceil(self.loc_patch.n_patch/self.batch_size).astype(int)):
    		batch_of_patch_index = const_array + self.batch_size*index
    		batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
    		
    		# get one batch_size
    		for n, selected_patch in enumerate(batch_of_patch_index):
    			X0[n, :,:,:,0] = self.loc_patch.__get_single_patch__without_padding_test__(X_pad, selected_patch)
    			
    		# predict the segmentation patches for the current batch
    		prediction1 = self.model1(X0, training = False)
    		prediction2 = self.model2(X0, training = False)
    		prediction3 = self.model3(X0, training = False)
    		prediction4 = self.model4(X0, training = False)

   		
    		# put the label back
    		for n, selected_patch in enumerate(batch_of_patch_index):
    			Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction1[n]), selected_patch) #Put single patch perform patch-wise addition
    			Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction2[n]), selected_patch)
    			Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction3[n]), selected_patch)
    			Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction4[n]), selected_patch)   
    	
    	Y0 = crop3D_hotEncoding(Y0, self.image_size, len(self.labels))
    	Y = restore_labels(Y0, self.labels)
    	
    	return Y

  	
# class BayesianPredict(object):
#     'run the model on test data'
# 
#     def __init__(self, model, image_size, patch_size=(32, 32, 32), labels=[1], T = 6):
#         'Initialization'
#         self.patch_size = patch_size
#         self.labels = labels
#         self.stride = [14, 14, 14]
#         self.image_size = np.asarray(image_size)
#         self.model = model
#         self.loc_patch = patch(self.image_size, patch_size, self.stride)
#         self.T = T
#         self.batch_size = 6
# 
#     def __run__(self, X):
#     	'test on one image each time'
#     
# 		# Pad the input image X if necessary
#     	if np.any(self.loc_patch.pad_width > [0, 0, 0]):
#     		X_pad = np.pad(X, mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
#     	else:
#     		X_pad = X
#     	   	
#     	   	
#     	output_size = np.append((self.loc_patch.size_after_pad), len(self.labels))
#     	output_size = np.append(self.T, output_size)
#     	
#     	Y0 = np.zeros(output_size,  dtype=np.float32)
#     	X0 = np.zeros([self.batch_size]+self.patch_size+[1])
#     	
#     	const_array = np.asarray(range(self.batch_size)) #Initialize: array with elements are iteratively equal to 0 until batch_size-1
#     	
#     	# Count the number of times each voxel will be visited
#     	count = np.zeros(np.append((self.loc_patch.size_after_pad), len(self.labels)), dtype = np.float32)
#     	onesPatch = np.ones(np.append((self.patch_size), len(self.labels)), dtype = np.float32)
#     	for index in range(np.ceil(self.loc_patch.n_patch/self.batch_size).astype(int)):
#     		batch_of_patch_index = const_array + self.batch_size*index
#     		batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
#    		
#     		for n, selected_patch in enumerate(batch_of_patch_index): #Add the patch to the counter
#     			count = self.loc_patch.__put_single_patch__(count, onesPatch, selected_patch) 
#     			
#     			  	
#     	for iPass in range(self.T):
#     	
#     		for index in range(np.ceil(self.loc_patch.n_patch/self.batch_size).astype(int)):
#     			batch_of_patch_index = const_array + self.batch_size*index
#     			batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
#     		
#     			# get one batch_size
#     			for n, selected_patch in enumerate(batch_of_patch_index):
#     				X0[n, :,:,:,0] = self.loc_patch.__get_single_patch__without_padding_test__(X_pad, selected_patch)
#     			
#     			# predict the segmentation patches for the current batch
#     			prediction = self.model(X0, training = False)
#    		
#     			# put the label back
#     			for n, selected_patch in enumerate(batch_of_patch_index):
#     				Y0[iPass] = self.loc_patch.__put_single_patch__(Y0[iPass], np.squeeze(prediction[n]), selected_patch) #Put single patch perform patch-wise addition
#     		
#     		
#     		Y0[iPass] = Y0[iPass] / count
#     	
#     	# Compute the mean of the predicted probability maps
#     	YProbaMap = np.mean(Y0, axis = 0)
#     	YProbaMap = crop3D_hotEncoding(YProbaMap, self.image_size, len(self.labels))		
# 
#     	Y = restore_labels(YProbaMap, self.labels)
#     	
#     	return Y, Y0

class Bayesian_autoContext_Predict(object):
    'run the model on test data'

    def __init__(self, model, image_size, patch_size=(32, 32, 32), labels=[1], T = 5):
        'Initialization'
        self.patch_size = patch_size
        # self.data_file = data_file
        self.labels = labels
        self.stride = [14,14, 14]
        self.image_size = np.asarray(image_size)
        self.model = model
        self.loc_patch = patch(self.image_size, patch_size, self.stride)
        self.T = T

    def __run__(self, X, uncertainty):
    	'test on one image each time'
    	
    	start_time_subject = time.time()
    
    	output_size = np.append((self.loc_patch.size_after_pad), len(self.labels))
    	output_size = np.append(self.T, output_size)
    	
    	Y0 = np.zeros(output_size,  dtype=np.float32)
    	batch_size = 6
    	X0 = np.zeros([batch_size]+self.patch_size+[1])
    	uncertainty0 = np.zeros([batch_size]+self.patch_size+[1])
    	
    	count = np.zeros(np.append((self.loc_patch.size_after_pad), len(self.labels)), dtype = np.float32)
    	
    	# Count the number of times each voxel will be visited
    	onesPatch = np.ones(np.append((self.patch_size), len(self.labels)), dtype = np.float32)
    	for index in range(np.ceil(self.loc_patch.n_patch/batch_size).astype(int)):
    		batch_of_patch_index = np.asarray(range(batch_size)) + batch_size*index
    		batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
   		
    		for n, selected_patch in enumerate(batch_of_patch_index):
    			count = self.loc_patch.__put_single_patch__(count, onesPatch, selected_patch) 
    			
    			  	
    	for iPass in range(self.T):
    	
    		for index in range(np.ceil(self.loc_patch.n_patch/batch_size).astype(int)):
    			batch_of_patch_index = np.asarray(range(batch_size)) + batch_size*index
    			batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
    		
    			# get one batch_size
    			for n, selected_patch in enumerate(batch_of_patch_index):
    				temp = self.loc_patch.__get_single_patch__(X, selected_patch)
    				X0[n] = temp[np.newaxis, ..., np.newaxis]
    				
    				temp_uncertainty = self.loc_patch.__get_single_patch__(uncertainty, selected_patch)
    				uncertainty0[n] = temp_uncertainty[np.newaxis, ..., np.newaxis]
    			
    			
    			# predict these patches
    			prediction = self.model.predict(X0, uncertainty0)
   		
    			# put the label back
    			for n, selected_patch in enumerate(batch_of_patch_index):
    				Y0[iPass] = self.loc_patch.__put_single_patch__(Y0[iPass], np.squeeze(prediction[n]), selected_patch)
    				
    		Y0[iPass] = Y0[iPass] / count	
    	
    	# Compute the mean of the predicted probability maps
    	YProbaMap = np.mean(Y0, axis = 0)
    	YProbaMap = crop3D_hotEncoding(YProbaMap, self.image_size, len(self.labels))		

    	Y = restore_labels(YProbaMap, self.labels)

    	end_time_subject = time.time()
    	
    	print('Prediction time for one subject is:' + str(end_time_subject - start_time_subject))
    	
    	return Y, Y0

class refined_UNet_predict(object):
    'run the model on test data'

    def __init__(self, model, image_size, patch_size=(32, 32, 32), labels=[1]):
        'Initialization'
        self.patch_size = patch_size
        # self.data_file = data_file
        self.labels = labels
        self.stride = [14, 14, 14]
        self.image_size = np.asarray(image_size)
        self.model = model
        self.loc_patch = patch(self.image_size, patch_size, self.stride)

    def __run__(self, X, uncertainty):
    	'test on one image each time'
    	
    	start_time_subject = time.time()
    
    	output_size = np.append((self.loc_patch.size_after_pad),len(self.labels))
    	Y0 = np.zeros(output_size,  dtype=np.float32)
    	batch_size = 7
    	X0 = np.zeros([batch_size]+self.patch_size+[1])
    	uncertainty0 = np.zeros([batch_size]+self.patch_size+[1])
    	
    	for index in range(np.ceil(self.loc_patch.n_patch/batch_size).astype(int)):
    		batch_of_patch_index = np.asarray(range(batch_size)) + batch_size*index
    		batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
    		
    		# get one batch_size
    		for n, selected_patch in enumerate(batch_of_patch_index):
    			tempImg = self.loc_patch.__get_single_patch__(X, selected_patch)
    			#temp = normlize_mean_std(temp)
    			X0[n] = tempImg[np.newaxis, ..., np.newaxis] # Why not write do this process in one sentences (why the use of newaxis is required) ?
    			
    			tempUncertainty = self.loc_patch.__get_single_patch__(uncertainty, selected_patch)
    			uncertainty0[n] = tempUncertainty[np.newaxis, ..., np.newaxis] # Why not write do this process in one sentences (why the use of newaxis is required) ?
    			
    		# predict these patches
#     		start_time = time.time()
    		prediction = self.model.predict([X0, uncertainty0])
#     		end_time = time.time()
#     		print('Prediction function (batch-wise) time:' + str(end_time - start_time))
    		
    		# put the label back
#     		start_time = time.time()
    		for n, selected_patch in enumerate(batch_of_patch_index):
    			Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction[n]), selected_patch)
    		
    	Y0 = crop3D_hotEncoding(Y0, self.image_size, len(self.labels))	
    	result = restore_labels(Y0, self.labels)

    	
    	# start_time = time.time()
#     	result = crop_pad3D(Y, self.image_size)
#     	end_time = time.time()
#     	print('Crop pad time:' + str(end_time-start_time))
    	
    	end_time_subject = time.time()
    	print('Prediction time for one subject is:' + str(end_time_subject - start_time_subject))
    	
    	return result

def test(x, model, image_size, patch_size, labels):
    prediction = np.zeros(x.shape)
    predictor = predict(model, image_size, patch_size, labels)
    for n in range(x.shape[0]):
        prediction[n] = predictor.__run__(x[n])
    return prediction


def bayesian_test(x, model, image_size, patch_size, labels):

    predictions = np.zeros(x.shape)
    probaMaps = np.zeros(np.append((x.shape), len(labels)))
    
    predictor = BayesianPredict(model, image_size, patch_size, labels)
    
    for n in range(x.shape[0]):
        predictions[n], probaMaps[n] = predictor.__run__(x[n])
        
    return predictions, probaMaps
    
def evaluate(x, y_true, model, image_size, patch_size, labels, ID, output_path='/home/axel/dev/neonatal_brain_segmentation/source/output/'):
    if not output_path:
        output_path = 'output/'
    n_subject = x.shape[0]
    predictor = predict(model, image_size, patch_size, labels)
    metric = 0.0
    np.set_printoptions(precision=3)
    for n in range(n_subject):
        y_pred = predictor.__run__(x[n])
        tmp = metrics.dice_multi_array(y_true[n], y_pred, labels)
        print(str(n)+': '+str(tmp))
        metric += tmp
        write_nii(y_pred, output_path+str(ID[n])+'test.nii')
    return metric/n_subject
