# lizhao U-Net 3D model core
from tensorflow.keras.layers import Conv3D, MaxPool3D, concatenate, Input, Dropout, PReLU, Conv3DTranspose, BatchNormalization
from tensorflow.keras.models import Model
# from keras_contrib.layers import CRF


def unet_core(x, filter_size=8, kernel_size=(3, 3, 3)):

    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    
    return x


def unet3d(patch_size, n_label):

    input_layer = Input(shape=patch_size)
    
    d1 = unet_core(input_layer, filter_size=64, kernel_size=(3, 3, 3))
    #l = MaxPool3D(strides=(2, 2, 2))(d1)
    l = Conv3D(filters=64, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d1)
    d2 = unet_core(l, filter_size=128, kernel_size=(3, 3, 3))
    l =  Conv3D(filters=128, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d2)
    d3 = unet_core(l, filter_size=256, kernel_size=(3, 3, 3))
    l = Conv3D(filters=256, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(d3)
    b = unet_core(l, filter_size=512, kernel_size=(3, 3, 3))
    b = Dropout(0.5)(b)

    l = Conv3DTranspose(filters=256, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(b)
    l = concatenate([l, d3], axis=-1)
    u3 = unet_core(l, filter_size=256, kernel_size=(3, 3, 3))
    l = Conv3DTranspose(filters=128, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(u3)
    l = concatenate([l, d2], axis=-1)
    u2 = unet_core(l, filter_size=128, kernel_size=(3, 3, 3))
    l = Conv3DTranspose(filters=64, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal')(u2)
    l = concatenate([l, d1], axis=-1)
    u1 = unet_core(l, filter_size=64, kernel_size=(3, 3, 3))

    output_layer = Conv3D(filters=n_label, kernel_size=(1, 1, 1), activation='softmax')(u1)
    # output_layer = CRF(n_label)
    model = Model(input_layer, output_layer)
    
    return model
    
    
def dense_block_aux(x, kernel_size_conv, growth_rate):

	x = BatchNormalization()(x)
	x = PReLU()(x)
	x = Conv3D(filters = growth_rate, kernel_size = kernel_size_conv, padding = 'same', kernel_initializer='he_normal')(x)
	
	
	return x


def dense_block(x, kernel_size_conv, growth_rate, nConv):
	
	feature_maps = []
	
	for iConv in range(nConv):
		 result_dense_block = dense_block_aux(x, kernel_size_conv, growth_rate)
		 x =  concatenate([x, result_dense_block], axis = -1)
		 feature_maps.append(result_dense_block)
	
	return concatenate(feature_maps,  axis = -1)

	
def transition_down_block(x, input_shape, kernel_size_conv, theta):
	
	x = BatchNormalization()(x)
	x = PReLU()(x)
# 	x = Conv3D(filters = int(input_shape * theta), kernel_size = kernel_size_conv, strides = [2, 2, 2], padding = 'same', kernel_initializer = 'he_normal')(x)
	x = Conv3D(filters = input_shape, kernel_size = kernel_size_conv, strides = [1, 1, 1], padding = 'same', kernel_initializer = 'he_normal')(x)
# 	x = Dropout(0.2)(x)
	x = MaxPool3D(strides=(2, 2, 2))(x)
	
	return x

def transition_up_block(x, input_shape, kernel_size_conv, theta):
	
# 	x = Conv3DTranspose(filters = int(input_shape * theta), kernel_size = kernel_size_conv, strides = [3, 3, 3], padding = 'same', kernel_initializer = 'he_normal')(x)
	x = Conv3DTranspose(filters = input_shape, kernel_size = kernel_size_conv, strides = [2, 2, 2], padding = 'same', kernel_initializer = 'he_normal')(x)
	
	return x
	

def tiramisu(patch_size, n_label):

    input_layer = Input(shape=patch_size)
    growthRate = 8				
    theta = 0.25

    
    x = Conv3D(filters = 42, kernel_size = (3, 3, 3), padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    x = BatchNormalization()(x)
    x = PReLU()(x)


    d1 = dense_block(x, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 3)
    c1 = concatenate([x, d1], axis = -1)  
    t1 = transition_down_block(c1, input_shape = c1.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('c1 shape is: ' + str(c1.get_shape().as_list()))
    print('t1 shape is: ' + str(t1.get_shape().as_list()))

    d2 = dense_block(t1, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 3)
    c2 = concatenate([t1, d2], axis = -1)
    t2 = transition_down_block(c2, input_shape = c2.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('c2 shape is: ' + str(c2.get_shape().as_list()))
    print('t2 shape is: ' + str(t2.get_shape().as_list()))
    
    
    d3 = dense_block(t2, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 4)
    c3 = concatenate([t2, d3], axis = -1)
    t3 = transition_down_block(c3, input_shape = c3.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('c3 shape is: ' + str(c3.get_shape().as_list()))
    print('t3 shape is: ' + str(t3.get_shape().as_list()))
    
    d4 = dense_block(t3, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 4)
	 
  
    t1 = transition_up_block(d4, input_shape = d4.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    c1 = concatenate([t1, d3], axis = -1)
    u1 = dense_block(c1, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 4)
    print('t1 up shape is: ' + str(t1.get_shape().as_list()))
    print('c1 up shape is: ' + str(c1.get_shape().as_list()))


    t2 = transition_up_block(u1, input_shape = u1.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    c2 = concatenate([t2, d2], axis = -1)  
    u2 = dense_block(c2, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 3)
    print('t2 up shape is: ' + str(t2.get_shape().as_list()))
    print('c2 up shape is: ' + str(c2.get_shape().as_list()))
    
    t3 = transition_up_block(u2, input_shape = u2.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    c3 = concatenate([t3, d1], axis = -1)  
    u3 = dense_block(c3, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 3)
    print('t3 up shape is: ' + str(t3.get_shape().as_list()))
    print('c3 up shape is: ' + str(c3.get_shape().as_list()))
    
        
    output_layer = Conv3D(filters = n_label, kernel_size = (1, 1, 1), activation = 'softmax')(u3)
    
    print('last layer shape is: ' + str(output_layer.get_shape().as_list()))
    
    model = Model(input_layer, output_layer)
    
    return model
  
def denseNet(patch_size, n_label):

    input_layer = Input(shape=patch_size)
    growthRate = 16			
    theta = 0.25

    
    x = Conv3D(filters = 64, kernel_size = (3, 3, 3), padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    x = BatchNormalization()(x)
    x = PReLU()(x)


    d1 = dense_block(x, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 3)
    c1 = concatenate([x, d1], axis = -1)  
    t1 = transition_down_block(c1, input_shape = c1.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('c1 shape is: ' + str(c1.get_shape().as_list()))
    print('t1 shape is: ' + str(t1.get_shape().as_list()))

    d2 = dense_block(t1, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 3)
    c2 = concatenate([t1, d2], axis = -1)
    t2 = transition_down_block(c2, input_shape = c2.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('c2 shape is: ' + str(c2.get_shape().as_list()))
    print('t2 shape is: ' + str(t2.get_shape().as_list()))
    
    
    d3 = dense_block(t2, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 4)
    c3 = concatenate([t2, d3], axis = -1)
    t3 = transition_down_block(c3, input_shape = c3.get_shape().as_list()[-1], kernel_size_conv = (2, 2, 2), theta = theta)
    print('c3 shape is: ' + str(c3.get_shape().as_list()))
    print('t3 shape is: ' + str(t3.get_shape().as_list()))
    
    d4 = dense_block(t3, kernel_size_conv = (3, 3, 3), growth_rate = growthRate, nConv = 5)
    
    #Up sampling path
    
    d2_up = Conv3DTranspose(filters = c2.get_shape().as_list()[-1], kernel_size = (2, 2, 2), strides = [2, 2, 2], padding = 'same', kernel_initializer = 'he_normal')(c2) 
    d3_up = Conv3DTranspose(filters = c3.get_shape().as_list()[-1], kernel_size = (2, 2, 2), strides = [4, 4, 4], padding = 'same', kernel_initializer = 'he_normal')(c3)
    d4_up = Conv3DTranspose(filters = d4.get_shape().as_list()[-1], kernel_size = (2, 2, 2), strides = [8, 8, 8], padding = 'same', kernel_initializer = 'he_normal')(d4)
    
    c = concatenate([d1, d2_up, d3_up, d4_up], axis = -1)
        
    output_layer = Conv3D(filters = n_label, kernel_size = (1, 1, 1), activation = 'softmax')(c)
    
    print('last layer shape is: ' + str(output_layer.get_shape().as_list()))
    
    model = Model(input_layer, output_layer)
    
    return model  
    
    