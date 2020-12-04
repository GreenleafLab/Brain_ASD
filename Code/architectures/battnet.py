import numpy as np
from kerasAC.metrics import *
import keras;
import pdb;
from keras.models import Model
from keras.layers import Input, Bidirectional
from keras.layers.merge import Add, Multiply, Dot
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Dense, Activation, Reshape
from keras.optimizers import Adam;
import keras.losses;
import keras.activations
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from keras.activations import softmax

def softMaxAxis1(x):
    return softmax(x,axis=1)
    

#note: functional model
#this is single-headed battnet (i.e. one attention head only) 
def getModelGivenModelOptionsAndWeightInits(args):
    #read in the arguments
    w0=args.w0
    w1=args.w1
    init_weights=args.init_weights
    ntasks=args.ntasks
    seed=args.seed
    
    np.random.seed(seed)
    K.set_image_data_format('channels_last')
    print(K.image_data_format())
    inputs=Input(shape=(1000,4))    
    gru_output=Bidirectional(GRU(512,dropout=0.3,input_dim=(1000,4),implementation=2,return_sequences=True))(inputs)
    
    
    #first output is attention
    attn1=Dense(1024)(gru_output)
    attn1=BatchNormalization(axis=-1)(attn1)
    attn1=Activation("relu")(attn1)
    attn1=Dropout(0.3)(attn1)

    attn1=Dense(1)(attn1)
    attn1=Activation("relu")(attn1)
    
    attn1_output=Activation(softMaxAxis1)(attn1)
    #add in fc 3 - 5
    added=Dot(axes=1)([gru_output,attn1_output])
    added=Reshape((1024,))(added)
    x=Dense(512)(added)
    x=BatchNormalization(axis=-1)(x)
    x=Activation("relu")(x)
    x=Dropout(0.3)(x)

    x=Dense(512)(x)
    x=BatchNormalization(axis=-1)(x)
    x=Activation("relu")(x)
    intermediate_out=Dropout(0.3)(x)

    #the output layer    
    outputs=Dense(ntasks,activation="sigmoid")(intermediate_out)
    
    model=Model(inputs=inputs,outputs=outputs)
    print(model.summary())
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("compiling!")
    if w0!=None:
        import kerasAC.custom_losses
        loss=kerasAC.custom_losses.get_weighted_binary_crossentropy(w0_weights=w0,w1_weights=w1)
    else:
        loss="binary_crossentropy"
    model.compile(optimizer=adam,loss=loss,metrics=[positive_accuracy,negative_accuracy,precision,recall])
    return model
