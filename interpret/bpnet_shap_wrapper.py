import pdb 
import argparse
import pickle
import tensorflow
from tensorflow.compat.v1.keras.backend import get_session
tensorflow.compat.v1.disable_v2_behavior()
import kerasAC 
from kerasAC.generators.tiledb_predict_generator import *
from kerasAC.tiledb_config import *
from kerasAC.interpret.deepshap import * 
from kerasAC.interpret.profile_shap import * 
from kerasAC.helpers.transform_bpnet_io import *
#load the model!
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from kerasAC.metrics import * 
from kerasAC.custom_losses import * 

def parse_args():
    parser=argparse.ArgumentParser(description="wrapper to make it easier to deepSHAP bpnet models")
    parser.add_argument("--ref_fasta")
    parser.add_argument("--chipseq",action='store_true',default=False) 
    parser.add_argument("--model_hdf5")
    parser.add_argument("--bed_regions")
    parser.add_argument("--bed_regions_center",choices=['summit','center'])
    parser.add_argument("--tdb_array")
    parser.add_argument("--chrom_sizes")
    parser.add_argument("--tdb_output_datasets",nargs="+")
    parser.add_argument("--tdb_input_datasets",nargs="+")
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--tdb_output_source_attribute",nargs="+",help="tiledb attribute for use in label generation i.e. fc_bigwig")
    parser.add_argument("--tdb_output_flank",nargs="+",help="flank around bin center to use in generating outputs")
    parser.add_argument("--tdb_output_aggregation",nargs="+",help="method for output aggregation; one of None, 'avg','max'")
    parser.add_argument("--tdb_output_transformation",nargs="+",help="method for output transformation; one of None, 'log','log10','asinh'")
    parser.add_argument("--tdb_input_source_attribute",nargs="+",help="attribute to use for generating model input, or 'seq' for one-hot-encoded sequence")
    parser.add_argument("--tdb_input_flank",nargs="+",help="length of sequence around bin center to use for input")
    parser.add_argument("--tdb_input_aggregation",nargs="+",help="method for input aggregation; one of 'None','avg','max'")
    parser.add_argument("--tdb_input_transformation",nargs="+",help="method for input transformation; one of None, 'log','log10','asinh'")
    parser.add_argument("--num_inputs",type=int)
    parser.add_argument("--num_outputs",type=int) 
    parser.add_argument("--out_pickle")
    parser.add_argument("--task_index",type=int)
    parser.add_argument("--num_threads",type=int)
    return parser.parse_args() 

def load_model_wrapper(args): 
    custom_objects={"recall":recall,
                        "sensitivity":recall,
                        "specificity":specificity,
                        "fpr":fpr,
                        "fnr":fnr,
                        "precision":precision,
                        "f1":f1,
                        "ambig_binary_crossentropy":ambig_binary_crossentropy,
                        "ambig_mean_absolute_error":ambig_mean_absolute_error,
                        "ambig_mean_squared_error":ambig_mean_squared_error,
                        "MultichannelMultinomialNLL":MultichannelMultinomialNLL}
    get_custom_objects().update(custom_objects)
    model=load_model(args.model_hdf5)
    return model

def get_generator(args):
    gen=TiledbPredictGenerator(ref_fasta=args.ref_fasta,
                               batch_size=args.batch_size,
                               bed_regions_center=args.bed_regions_center,
                               bed_regions=args.bed_regions,
                               tdb_partition_thresh_for_upsample=None,
                               tdb_partition_attribute_for_upsample=None,
                               tdb_partition_datasets_for_upsample=None,
                               tdb_output_datasets=args.tdb_output_datasets,
                               tdb_input_datasets=args.tdb_input_datasets,
                               tdb_array=args.tdb_array,
                               chrom_sizes=args.chrom_sizes,
                               tdb_input_flank=args.tdb_input_flank,
                               tdb_input_source_attribute=args.tdb_input_source_attribute,
                               tdb_input_aggregation=args.tdb_input_aggregation,
                               tdb_input_transformation=args.tdb_input_transformation,
                               tdb_output_source_attribute=args.tdb_output_source_attribute,
                               tdb_output_flank=args.tdb_output_flank,
                               tdb_output_aggregation=args.tdb_output_aggregation,
                               tdb_output_transformation=args.tdb_output_transformation,
                               num_inputs=args.num_inputs,
                               num_outputs=args.num_outputs,
                               upsample_ratio=None,
                               tdb_ambig_attribute=None,
                               tdb_config=get_default_config(),
                               tdb_ctx=tiledb.Ctx(config=get_default_config()),
                               num_threads=args.num_threads)
    return gen

def get_interpretations(gen, model, count_explainer, prof_explainer,task_index,chipseq):
    label_prof_dict={} 
    label_count_dict={} 
    pred_prof_dict={} 
    pred_count_dict={} 
    profile_shap_dict={}
    count_shap_dict={}
    seq_dict={}
    length_gen=len(gen)
    for i in range(length_gen): 
        print(str(i)+'/'+str(length_gen))
        X,y,coords=gen[i]
        coords=[[entry.decode('utf8')  for entry in coord] for coord in coords]
        preds=model.predict(X)

        pred_prob=get_probability_track_from_bpnet(preds[0][:,:,task_index])
        label_prob=get_probability_label_track(y[0][:,:,task_index])
        
        label_sum=y[1][:,task_index]
        pred_sum=preds[1][:,task_index]
        #chipseq will have control profile, control counts; ATAC/histone will not
        seq_input=X[0]
        if chipseq is True:
            control_profile=X[1]
            control_counts=X[2]
            count_explanations=count_explainer.shap_values([seq_input,control_counts])[0]
        else:
            control_profile=None
            count_explanations=count_explainer.shap_values(X)[0]
        profile_explanations=prof_explainer(seq_input, control_profile)
        #print(str(len(profile_explanations)))
        #print(str(profile_explanations[0].shape))
        #print(str(profile_explanations[1].shape))
        #print(str(len(count_explanations)))
        #print(str(count_explanations[0].shape))
        #print(str(count_explanations[1].shape))

        #get explanations relative to input sequence, the bias track is in index 1
        profile_explanations=profile_explanations[0]
        count_explanations=count_explanations[0]        
        #store outputs in dictionary 
        for coord_index in range(len(coords)): 
            cur_coord=coords[coord_index][0:2]
            cur_coord[1]=int(cur_coord[1])
            cur_coord=tuple(cur_coord)
            label_prof_dict[cur_coord]=label_prob[coord_index]
            label_count_dict[cur_coord]=label_sum[coord_index]
            pred_prof_dict[cur_coord]=pred_prob[coord_index]
            pred_count_dict[cur_coord]=pred_sum[coord_index]
            profile_shap_dict[cur_coord]=profile_explanations[coord_index,:]
            count_shap_dict[cur_coord]=count_explanations[coord_index,:]
            seq_dict[cur_coord]=X[0][coord_index]    
    return label_prof_dict, label_count_dict,pred_prof_dict,pred_count_dict, profile_shap_dict, count_shap_dict, seq_dict

def main():
    args=parse_args()
    gen=get_generator(args)
    print("created generator")
    #load the model
    model=load_model_wrapper(args)
    print("loaded model")    
    if args.chipseq is True:
        create_background_counts=create_background_counts_chip
        model_wrapper_for_counts=([model.input[0],model.input[2]],model.outputs[1][:,args.task_index:args.task_index+1])
    else:
        create_background_counts=create_background_atac 
        model_wrapper_for_counts=(model.input, model.outputs[1][:,task_index:task_index+1])    
    count_explainer=shap.DeepExplainer(model_wrapper_for_counts,data=create_background_counts,combine_mult_and_diffref=combine_mult_and_diffref_1d)
    print("got count explainer") 
    prof_explainer = create_explainer(model,ischip=args.chipseq,task_index=args.task_index)
    print("got profile explainer")
    label_prof_dict, label_count_dict,pred_prof_dict,pred_count_dict, profile_shap_dict, count_shap_dict, seq_dict=get_interpretations(gen,model, count_explainer,prof_explainer,args.task_index,args.chipseq)
    print("finished with interpretations")
    #save the dictionaries to disk! 
    
    outputs={} 
    outputs['label_prof']=label_prof_dict
    outputs['label_sum']=label_count_dict
    outputs['pred_prof']=pred_prof_dict
    outputs['pred_sum']=pred_count_dict
    outputs['profile_shap']=profile_shap_dict 
    outputs['count_shap']=count_shap_dict 
    outputs['seq']=seq_dict 
    pickle.dump(outputs,open(args.out_pickle, "wb" ) )


if __name__=="__main__":
    main()
    
