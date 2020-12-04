import numpy as np ;
from keras.constraints import max_norm
from kerasAC.metrics import *
from kerasAC.custom_losses import get_weighted_binary_crossentropy, get_ambig_binary_crossentropy
from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1


def getModelGivenModelOptionsAndWeightInits(args):
    seed=args.seed
    w0=args.w0
    w1=args.w1
    ntasks=args.ntasks
    init_weights=args.init_weights
    
    np.random.seed(seed)
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

    if (init_weights!=None):
        #load the weight initializations 
        data=np.load(init_weights);
        model=Sequential()
        model.add(Conv2D(filters=300,kernel_size=(1,19),input_shape=(1,1000,4),weights=[data['0.Conv/weights:0'],np.zeros(300,)],padding="same"))
        model.add(BatchNormalization(axis=-1,weights=[data['2.BatchNorm/gamma:0'],data['1.BatchNorm/beta:0'],np.zeros(300,),np.zeros(300,)]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,3)))
        model.add(Conv2D(filters=200,kernel_size=(1,11),weights=[data['3.Conv_1/weights:0'],np.zeros(200,)],padding="same"))
        model.add(BatchNormalization(axis=-1,weights=[data['5.BatchNorm_1/gamma:0'],data['4.BatchNorm_1/beta:0'],np.zeros(200,),np.zeros(200,)]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,4)))
        model.add(Conv2D(filters=200,kernel_size=(1,7),weights=[data['6.Conv_2/weights:0'],np.zeros(200,)],padding="same"))
        model.add(BatchNormalization(axis=-1,weights=[data['8.BatchNorm_2/gamma:0'],data['7.BatchNorm_2/beta:0'],np.zeros(200,),np.zeros(200,)]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,4)))

        model.add(Flatten())
        model.add(Dense(1000,weights=[data['9.fc0/fully_connected/weights:0'],np.zeros(1000,)]))
        model.add(BatchNormalization(axis=1,weights=[data['11.fc0/BatchNorm/gamma:0'],data['10.fc0/BatchNorm/beta:0'],np.zeros(1000,),np.zeros(1000,)]))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(1000,weights=[data['12.fc1/fully_connected/weights:0'],np.zeros(1000,)]))
        model.add(BatchNormalization(axis=1,weights=[data['14.fc1/BatchNorm/gamma:0'],data['13.fc1/BatchNorm/beta:0'],np.zeros(1000,),np.zeros(1000,)]))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(ntasks))
        model.add(Activation("sigmoid"))

    else:
        model=Sequential()
        model.add(Conv2D(filters=300,kernel_size=(1,19),input_shape=(1,1000,4)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,3)))

        model.add(Conv2D(filters=200,kernel_size=(1,11)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,4)))

        model.add(Conv2D(filters=200,kernel_size=(1,7)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,4)))

        model.add(Flatten())
        model.add(Dense(1000))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(1000))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(ntasks))
        model.add(Activation("sigmoid"))
        
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("compiling!")
    if w0!=None:
        loss=get_weighted_binary_crossentropy(w0_weights=w0,w1_weights=w1)
    else:
        loss=ambig_binary_crossentropy 
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=[recall, specificity, fpr, fnr, precision, f1])
    return model
