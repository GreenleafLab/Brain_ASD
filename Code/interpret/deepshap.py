from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
import tensorflow as tf
import numpy as np


def deepshap_wrapper(inputs):
    X=inputs[0]
    explainer=inputs[1]
    return explainer.shap_values(X,progress_message=10)

def create_background(inputs, bg_size=10, seed=1234):
    input_seq=inputs[0]
    if len(inputs)==2:
        input_seq_bg = [np.empty((bg_size,) + input_seq.shape),np.asarray(bg_size*[inputs[1]])]
    elif len(inputs)==1:
        input_seq_bg = [np.empty((bg_size,) + input_seq.shape)]
    rng = np.random.RandomState(seed)
    for i in range(bg_size):
        input_seq_shuf = dinuc_shuffle(np.squeeze(input_seq), rng=rng)
        input_seq_bg[0][i] = np.expand_dims(input_seq_shuf,axis=0)
    return input_seq_bg


def combine_mult_and_diffref_1d(mult, orig_inp, bg_data):
    to_return = []
    for l in [0]:
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:,i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference*mult[l]
            projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1) 
        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))
    if len(orig_inp)>1: 
        to_return.append(np.zeros_like(orig_inp[1]))
    return to_return

def combine_mult_and_diffref_2d(mult, orig_inp, bg_data):
    to_return = []
    for l in [0]:
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==3
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:,:,i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None,:,:,:]-bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference*mult[l]
            projected_hypothetical_contribs[:,:,:,i] = np.sum(hypothetical_contribs,axis=-1) 
        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))
    if len(orig_inp)>1: 
        to_return.append(np.zeros_like(orig_inp[1]))
    return to_return

def create_explainer(model,shuffle_func,target_layer,combine_mult_and_diffref,task_index):
    model_wrapper=(model.input, model.layers[target_layer].output[:,task_index])
    explainer=shap.DeepExplainer(model_wrapper,
                                 data=shuffle_func,
                                 combine_mult_and_diffref=combine_mult_and_diffref)
    return explainer


