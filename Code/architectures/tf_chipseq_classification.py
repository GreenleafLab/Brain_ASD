import numpy as np ;
from keras.constraints import max_norm
from kerasAC.metrics import *
from kerasAC.custom_losses import get_weighted_binary_crossentropy, get_ambig_binary_crossentropy
from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1
import keras;
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop;
import keras.losses;
from keras.constraints import maxnorm;
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2    
from keras import backend as K
K.set_image_data_format('channels_last')
print(K.image_data_format())

def getModelGivenModelOptionsAndWeightInits(args):
    #read in the args
    seed=args.seed
    ntasks=args.ntasks
    w0=args.w0
    w1=args.w1
    
    np.random.seed(seed)
    model=Sequential()
    model.add(Conv2D(filters=50,kernel_size=(1,15),padding="same", kernel_constraint=max_norm(7.0,axis=-1),input_shape=(1,1000,4)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))   
    model.add(Conv2D(filters=50,kernel_size=(1,15),padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))    
    model.add(Conv2D(filters=50,kernel_size=(1,13),padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(1,40)))    
    model.add(Flatten())
    model.add(Dense(50))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))    
    model.add(Dense(ntasks))
    model.add(Activation("sigmoid"))
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("compiling!")
    if w0!=None:
        loss=get_weighted_binary_crossentropy(w0_weights=w0,w1_weights=w1)
    else:
        loss=get_ambig_binary_crossentropy() 
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=[recall, specificity, fpr, fnr, precision, f1])
    return model
