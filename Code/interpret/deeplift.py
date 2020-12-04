import deeplift 

def deeplift_wrapper(inputs):
    X=inputs[0]
    score_func=inputs[1]
    task_idx=inputs[2]
    num_refs_per_seq=inputs[3]
    reference=inputs[4]
    return deeplift_batch(score_func,X,task_idx,num_refs_per_seq,reference)

def deeplift_zero_ref(X,score_func,batch_size=200,task_idx=0):        
    # use a 40% GC reference
    input_references = [np.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]]
    # get deeplift scores
    
    deeplift_scores = score_func(
        task_idx=task_idx,
        input_data_list=[X],
        batch_size=batch_size,
        progress_update=None,
        input_references_list=input_references)
    return deeplift_scores

def deeplift_gc_ref(X,score_func,batch_size=200,task_idx=0):        
    # use a 40% GC reference
    input_references = [np.array([0.3, 0.2, 0.2, 0.3])[None, None, None, :]]
    # get deeplift scores
    
    deeplift_scores = score_func(
        task_idx=task_idx,
        input_data_list=[X],
        batch_size=batch_size,
        progress_update=None,
        input_references_list=input_references)
    return deeplift_scores

def deeplift_shuffled_ref(X,score_func,batch_size=200,task_idx=0,num_refs_per_seq=10):
    deeplift_scores=score_func(task_idx=task_idx,input_data_sequences=X,num_refs_per_seq=num_refs_per_seq,batch_size=batch_size)
    return deeplift_scores

def get_deeplift_scoring_function(model,target_layer_idx=-2,task_idx=0, reference="shuffled_ref", sequential=True):
    """
    Arguments: 
        model -- a string containing the path to the hdf5 exported model 
        target_layer_idx -- should be -2 for classification; -1 for regression 
        reference -- one of 'shuffled_ref','gc_ref','zero_ref'
    Returns:
        deepLIFT scoring function 
    """
    from deeplift.conversion import kerasapi_conversion as kc
    deeplift_model = kc.convert_model_from_saved_files(model,verbose=False)

    #get the deeplift score with respect to the logit 
    if(sequential):
        score_func = deeplift_model.get_target_contribs_func(
             find_scores_layer_idx=task_idx,
             target_layer_idx=target_layer_idx)
    else:
        input_name = deeplift_model.get_input_layer_names()[0]
        target_layer_name = list(deeplift_model.get_name_to_layer().keys())[target_layer_idx]
        multipliers_func = deeplift_model.get_target_multipliers_func(input_name, target_layer_name)
        score_func = deeplift.util.get_hypothetical_contribs_func_onehot(multipliers_func)
    
    if reference=="shuffled_ref":
        from deeplift.util import get_shuffle_seq_ref_function
        from deeplift.dinuc_shuffle import dinuc_shuffle        
        score_func=get_shuffle_seq_ref_function(
            score_computation_function=score_func,
            shuffle_func=dinuc_shuffle,
            one_hot_func=None)
    return score_func


def deeplift_batch(score_func,X,task_idx,num_refs_per_seq,reference):
    batch_size=X.shape[0]
    if reference=="shuffled_ref":
        deeplift_scores_batch=deeplift_shuffled_ref(X,score_func,batch_size,task_idx,num_refs_per_seq)
    elif reference=="gc_ref":
        deeplift_scores_batch=deeplift_gc_ref(X,score_func,batch_size,task_idx)
    elif reference=="zero_ref":
        deeplift_scores_batch=deeplift_zero_ref(X,score_func,batch_size,task_idx)
    else:
        raise Exception("supported DeepLIFT references are 'shuffled_ref','gc_ref', 'zero_ref'")
    print("done with batch")
    #Project onto the base that's actually present
    deeplift_scores_batch = np.sum(deeplift_scores_batch, axis=-1)[:,:,:,None]*X
    return deeplift_scores_batch

