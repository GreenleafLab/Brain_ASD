import numpy as np
from kerasAC.metrics import *
import keras;
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import Add 
from keras.optimizers import Adadelta, SGD, RMSprop;
import keras.losses;
from keras.constraints import maxnorm;
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras import backend as K


def l1_block(x):
    tmp=Conv2D(filters=300,kernel_size=(1,9),padding="same")(x)
    tmp=BatchNormalization(axis=-1)(tmp)
    tmp=Activation('relu')(tmp)
    tmp=Conv2D(filters=300,padding="same",kernel_size=(1,7))(tmp)
    tmp=BatchNormalization(axis=-1)(tmp)
    added=Add()([x,tmp])
    out=Activation('relu')(added)
    return out

    
#note: functional model
def getModelGivenModelOptionsAndWeightInits(args):
    np.random.seed(args.seed)
    w0=args.w0
    w1=args.w1
    ntasks=args.ntasks
    
    K.set_image_data_format('channels_last')
    print(K.image_data_format())


    inputs=Input(shape=(1,1000,4))
    x=Conv2D(filters=300,kernel_size=(1,11))(inputs)
    x=BatchNormalization(axis=-1)(x)
    x=Activation('relu')(x)

    #add 2 x 1st resnet blocks
    x=l1_block(x)
    x=MaxPooling2D(pool_size=(1,4))(x)
    
    #Flatten
    x=Flatten()(x)

    #fully connected 1
    x=Dense(1000)(x)
    x=BatchNormalization(axis=-1)(x)
    x=Activation("relu")(x)
    x=Dropout(0.3)(x)

    #fully conected 2
    x=Dense(1000)(x)
    x=BatchNormalization(axis=-1)(x)
    x=Activation("relu")(x)
    x=Dropout(0.3)(x)
    
    #the output layer

    x=Dense(ntasks)(x)
    outputs=Activation("sigmoid")(x)
    
    model=Model(inputs=inputs,outputs=outputs)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("compiling!")
    import kerasAC.custom_losses
    if w0!=None:
        loss=kerasAC.custom_losses.get_weighted_binary_crossentropy(w0_weights=w0,w1_weights=w1)
    else:
        loss="binary_crossentropy"
    model.compile(optimizer=adam,loss=loss,metrics=[positive_accuracy,negative_accuracy,precision,recall])
    return model
