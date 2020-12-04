import numpy as np ;
from kerasAC.metrics import *
from kerasAC.custom_losses import *
def getModelGivenModelOptionsAndWeightInits(args):
    #get the arguments
    seed=args.seed
    w0=args.w0
    w1=args.w1
    ntasks=args.num_tasks
    
    np.random.seed(seed)
    import keras;
    from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
    from keras.layers.convolutional import Conv1D, MaxPooling1D
    from keras.optimizers import Adadelta, SGD, RMSprop;
    import keras.losses;
    from keras.constraints import maxnorm;
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l1, l2
    from keras import backend as K
    from keras.layers import Input, Concatenate 
    from keras.models import Model

    padding = "same"
    K.set_image_data_format('channels_last')
    print(K.image_data_format())
    seq = Input(shape=(1000,4),name='data/genome_data_dir')
    gc=Input(shape=(1,))

    x = Conv1D(filters=300,kernel_size=19,input_shape=(1000,4),padding=padding, name='conv1d_1')(seq)
    x = BatchNormalization(axis=-1,name='batch_normalization_1')(x)
    x = Activation('relu',name='activation_1')(x)
    x = MaxPooling1D(pool_size=3,name='max_pooling1d_1')(x)

    x = Conv1D(filters=200,kernel_size=11,padding=padding, name='conv1d_2')(x)
    x = BatchNormalization(axis=-1,name='batch_normalization_2')(x)
    x = Activation('relu',name='activation_2')(x)
    x = MaxPooling1D(pool_size=4,name='max_pooling1d_2')(x)

    x = Conv1D(filters=200,kernel_size=7,padding=padding, name='conv1d_3')(x)
    x = BatchNormalization(axis=-1,name='batch_normalization_3')(x)
    x = Activation('relu',name='activation_3')(x)
    x = MaxPooling1D(pool_size=4,name='max_pooling1d_3')(x)

    x = Flatten(name='flatten_1')(x)
    x = Dense(1000,name='dense_1')(x)
    x = BatchNormalization(axis=-1,name='batch_normalization_4')(x)
    x = Activation('relu',name='activation_4')(x)
    x = Dropout(0.3,name='dropout_1')(x)

    x = Dense(1000,name='dense_2')(x)
    x = BatchNormalization(axis=-1,name='batch_normalization_5')(x)
    x = Activation('relu',name='activation_5')(x)
    x = Dropout(0.3,name='dropout_2')(x)

    #add in the gc content
    added=Concatenate(axis=-1)([x,gc])
    y = Dense(ntasks,name='final_dense'+str(ntasks))(added)
    outputs = Activation("sigmoid",name='activation_6')(y)
    model = Model(inputs = [seq, gc], outputs = outputs)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("compiling!")
    loss=ambig_binary_crossentropy    
    model.compile(optimizer=adam,loss=loss)        
    return model
