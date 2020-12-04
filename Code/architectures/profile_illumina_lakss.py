
input_width=13000 
input_dimension=4
number_of_convolutions=11
filters=[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]
filter_dim=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
dilation=[1, 1, 5, 5, 25, 25, 125, 125, 153,160, 625]
activations='relu'
bn_true=True



def BatchNormalization_mod(conv, bn_flag=True):
    from keras.layers.normalization import BatchNormalization
    if bn_flag:
        return BatchNormalization()(conv)
    else:
        return conv


def res_block(conv,num_filter,f_width,act,d_rate,i,bn_true=True):
    crop_id=Cropping1D(d_rate*(f_width-1))(conv)
    conv1 = BatchNormalization_mod(conv,bn_true)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv1D(num_filter,f_width,dilation_rate=d_rate,padding="valid",name='conv_'+str(i)+'_a')(conv1)
    conv1 = BatchNormalization_mod(conv1,bn_true)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv1D(num_filter,f_width,dilation_rate=d_rate,padding="valid",name='conv_'+str(i)+'_b')(conv1)
    return keras.layers.Add()([conv1, crop_id])
    #return conv1


def get_conv1d(num_filter,f_width,d_rate,s_rate,layer_name,conv,layer_activation,layer_padding,bn_true=True):
    conv = BatchNormalization_mod(conv,bn_true)
    conv = Activation(layer_activation)(conv)
    conv = Conv1D(num_filter,f_width,dilation_rate=d_rate,strides=s_rate,padding=layer_padding,name=layer_name)(conv)
    return conv



def build1d_model_residual(input_width,input_dimension,number_of_convolutions,filters,filter_dim,dilation,activations,bn_true=True,max_flag=True):
    import tensorflow as tf
    import keras
    from keras import backend as K
    from keras.layers.pooling import GlobalMaxPooling1D,MaxPooling2D,MaxPooling1D
    from keras.models import Sequential,Model
    from keras.layers import Dense,Activation,Dropout,Flatten,Reshape,Input, Embedding, LSTM, Dense,Concatenate
    from keras.layers.convolutional import Conv1D,Conv2D
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l1,l2
    from keras.optimizers import SGD,RMSprop,Adam
    from sklearn.metrics import average_precision_score
    input1=Input(shape=(input_width,4), name='sequence')
    conv=get_conv1d(32,1,1,1,'upsampling',input1,'relu','same')
    for i in range(0,number_of_convolutions):
            conv = res_block(conv,filters[i],filter_dim[i],activations,dilation[i],i,bn_true)
    output=get_conv1d(1,1,1,1,'dnase',conv,'relu','same')
    #output=Conv1D(1,1,activation='relu',name='dnase')(conv)
    model = Model(input=[input1],output=[output])
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    return model

