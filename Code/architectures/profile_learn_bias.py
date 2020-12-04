import pdb 
import numpy as np ;
from keras.backend import int_shape
from sklearn.metrics import average_precision_score
from kerasAC.metrics import * 
from kerasAC.custom_losses import *

import keras;

#import the various keras layers 
from keras.layers import Dense,Activation,Dropout,Flatten,Reshape,Input, Concatenate, Cropping1D, Add
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D,MaxPooling1D,GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam
from keras.constraints import maxnorm;
from keras.regularizers import l1, l2    

from keras.models import Model

def get_model_param_dict(param_file):
    '''
    param_file has 2 columns -- param name in column 1, and param value in column 2
    '''
    params={}
    if param_file is None:
        return  params
    for line in open(param_file,'r').read().strip().split('\n'):
        tokens=line.split('\t')
        params[tokens[0]]=tokens[1]
    return params 

def getModelGivenModelOptionsAndWeightInits(args):
    #default params (can be overwritten by providing model_params file as input to the training function)
    filters=1
    conv1_kernel_size=6
    control_smoothing=[1, 50]
    counts_loss_weight=1
    profile_loss_weight=1
    
    model_params=get_model_param_dict(args.model_params)
    if 'filters' in model_params:
        filters=int(model_params['filters'])
    if 'conv1_kernel_size' in model_params:
        conv1_kernel_size=int(model_params['conv1_kernel_size'])
    if 'counts_loss_weight' in model_params:
        counts_loss_weight=float(model_params['counts_loss_weight'])
    if 'profile_loss_weight' in model_params:
        profile_loss_weight=float(model_params['profile_loss_weight'])

    print("params:")
    print("filters:"+str(filters))
    print("conv1_kernel_size:"+str(conv1_kernel_size))
    print("counts_loss_weight:"+str(counts_loss_weight))
    print("profile_loss_weight:"+str(profile_loss_weight))
    
    #read in arguments
    seed=int(args.seed)
    init_weights=args.init_weights 
    sequence_flank=int(args.tdb_input_flank[0])
    num_tasks=int(args.num_tasks)
    
    seq_len=2*sequence_flank
    out_flank=int(args.tdb_output_flank[0])
    out_pred_len=2*out_flank
    print(seq_len)
    print(out_pred_len)
    #define inputs
    inp = Input(shape=(seq_len, 4),name='sequence')    

    # first convolution without dilation
    first_conv = Conv1D(filters,
                        kernel_size=conv1_kernel_size,
                        padding='valid', 
                        activation='relu',
                        name='1st_conv')(inp)

    profile_out_prebias_shape =int_shape(first_conv)
    cropsize = int(profile_out_prebias_shape[1]/2)-int(out_pred_len/2)
    if profile_out_prebias_shape[1]%2==0:
        crop_left=cropsize
        crop_right=cropsize
    else:
        crop_left=cropsize
        crop_right=cropsize+1
    print(crop_left)
    print(crop_right)
    profile_out_prebias = Cropping1D((crop_left,crop_right),
                                     name='prof_out_crop2match_output')(first_conv)
    profile_out = Conv1D(filters=num_tasks,
                         kernel_size=1,
                         name="profile_predictions")(profile_out_prebias)
    gap_combined_conv = GlobalAveragePooling1D(name='gap')(first_conv)
    count_out = Dense(num_tasks, name="logcount_predictions")(gap_combined_conv)
    model=Model(inputs=[inp],outputs=[profile_out,
                                     count_out])
    print("got model") 
    model.compile(optimizer=Adam(),
                    loss=[MultichannelMultinomialNLL(1),'mse'],
                    loss_weights=[profile_loss_weight,counts_loss_weight])
    print("compiled model")
    return model 


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="view model arch")
    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--init_weights",default=None)
    parser.add_argument("--tdb_input_flank",nargs="+",default=[673])
    parser.add_argument("--tdb_output_flank",nargs="+",default=[500])
    parser.add_argument("--num_tasks",type=int,default=1)
    parser.add_argument("--model_params",default=None) 
    args=parser.parse_args()
    model=getModelGivenModelOptionsAndWeightInits(args)
    print(model.summary())
    pdb.set_trace() 
    
