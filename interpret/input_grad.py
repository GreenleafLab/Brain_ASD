#import keras functions
import keras 
from keras.models import Model 

def input_grad_wrapper(inputs):
    X=inputs[0]
    input_grad_function=inputs[1]
    input_to_use=inputs[2]
    return input_grad(input_grad_function,X,input_to_use=input_to_use)


#Careful! Gradientxinput is summed across tasks, there is no support in tensorflow for calculating the per-task gradient
#(see thread here: https://github.com/tensorflow/tensorflow/issues/4897) 
def get_input_grad_function(model,target_layer_idx=-2):
    print("WARNING: this function provides aggregated gradients across tasks. Not recommended for multi-tasked models")
    from keras import backend as K
    fn = K.function(model.inputs, K.gradients(model.layers[target_layer_idx].output, model.inputs))
    return fn

def input_grad(input_grad_function,X,input_to_use=0):
    return input_grad_function(X)[input_to_use]

