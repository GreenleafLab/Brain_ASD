import pandas as pd
import numpy as np 
from scipy.special import softmax,logit,softmax
from sklearn.preprocessing import normalize

def get_count_track_from_bpnet(profile_preds,count_preds):
    #get counts profile & softmax 
    profile_softmax=softmax(profile_preds,axis=1)
    return profile_softmax*np.expand_dims(count_preds,axis=1)

def get_probability_track_from_bpnet(profile_preds):
    return softmax(profile_preds,axis=1)

def get_logit_label_track(profile_labels,pseudocount=1e-4):
    return logit(profile_labels+pseudocount)

def get_probability_label_track(labels):
    return normalize(labels,norm='l1')

def get_model_outputs_to_plot(preds,coords=None):
    #label sum
    labels_sum=np.array(preds[('lab_1')]) 
    #label counts
    labels_profile_counts=np.array(preds[('lab_0')])
    #prediction sum
    predictions_sum=np.array(preds[('pred_1')])
    #prediction logits
    predictions_profile_logits=np.array(preds[('pred_0')])
    if coords is not None:
        labels_sum=labels_sum[coords]
        labels_profile_counts=labels_profile_counts[coords]
        predictions_sum=predictions_sum[coords]
        predictions_profile_logits=predictions_profile_logits[coords]
        
    #label prob
    labels_profile_prob=get_probability_label_track(labels_profile_counts)
    #label logits
    labels_profile_logits=get_logit_label_track(labels_profile_prob)
    #prediction prob
    predictions_profile_prob=get_probability_track_from_bpnet(predictions_profile_logits)
    #prediction counts
    predictions_profile_counts=get_count_track_from_bpnet(predictions_profile_logits,predictions_sum)
    #delta in logit space
    delta_logits=labels_profile_logits - predictions_profile_logits 
    #delta in probability space 
    delta_prob=get_probability_track_from_bpnet(delta_logits)
    return {'labels_counts':labels_profile_counts,
            'labels_logits':labels_profile_logits,
            'labels_prob':labels_profile_prob,
            'predictions_counts':predictions_profile_counts,
            'predictions_logits':predictions_profile_logits,
            'predictions_prob':predictions_profile_prob,
            'delta_logits':delta_logits,
            'delta_prob':delta_prob,
            'labels_sum':labels_sum,
            'predictions_sum':predictions_sum}


