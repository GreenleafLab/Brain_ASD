import tensorflow as tf
from .splits import *
from .config import args_object_from_args_dict
from .train import *
from .predict_hdf5 import *
from .interpret import *
from .performance_metrics.performance_metrics import * 
import argparse
import pdb

def parse_args():
    parser=argparse.ArgumentParser(description='Provide model files  & a dataset, get model predictions')
    parser.add_argument("--model_prefix",help="output model file that is generated at the end of training (in hdf5 format)")
    parser.add_argument("--assembly")
    parser.add_argument("--splits",nargs="+",default=None,type=int)
    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--use_multiprocessing",action='store_true',default=False)
    parser.add_argument("--multi_gpu",action='store_true',default=False)
    input_data_path=parser.add_argument_group("input_data_path")
    input_data_path.add_argument("--index_data_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels")
    input_data_path.add_argument("--index_train_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels for the training split")
    input_data_path.add_argument("--index_valid_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels for the validation split")
    input_data_path.add_argument("--index_tasks",nargs="*",default=None)
    input_data_path.add_argument("--input_data_path",nargs="+",default=None,help="seq or path to seqdataloader hdf5")
    input_data_path.add_argument("--input_train_path",nargs="+",default=None,help="seq or seqdataloader hdf5")
    input_data_path.add_argument("--input_valid_path",nargs="+",default=None,help="seq or seqdataloader hdf5")
    input_data_path.add_argument("--output_data_path",nargs="+",default=None,help="path to seqdataloader hdf5")
    input_data_path.add_argument("--output_train_path",nargs="+",default=None,help="seqdataloader hdf5")
    input_data_path.add_argument("--output_valid_path",nargs="+",default=None,help="seqdataloader hdf5")    
    input_data_path.add_argument('--variant_bed',default=None)
    input_data_path.add_argument('--ref_fasta')

    tiledbgroup=parser.add_argument_group("tiledb")
    tiledbgroup.add_argument("--tdb_outputs",nargs="+")
    tiledbgroup.add_argument("--tdb_output_source_attribute",nargs="+",default="fc_bigwig",help="tiledb attribute for use in label generation i.e. fc_bigwig")
    tiledbgroup.add_argument("--tdb_output_flank",nargs="+",type=int,help="flank around bin center to use in generating outputs")
    tiledbgroup.add_argument("--tdb_output_aggregation",nargs="+",default=None,help="method for output aggreagtion; one of None, 'avg','max'")
    tiledbgroup.add_argument("--tdb_output_transformation",nargs="+",default=None,help="method for output transformation; one of None, 'log','log10','asinh'")    
    tiledbgroup.add_argument("--tdb_inputs",nargs="+")
    tiledbgroup.add_argument("--tdb_input_source_attribute",nargs="+",help="attribute to use for generating model input, or 'seq' for one-hot-encoded sequence")
    tiledbgroup.add_argument("--tdb_input_flank",nargs="+",type=int,help="length of sequence around bin center to use for input")
    tiledbgroup.add_argument("--tdb_input_aggregation",nargs="+",default=None,help="method for input aggregation; one of 'None','avg','max'")
    tiledbgroup.add_argument("--tdb_input_transformation",nargs="+",default=None,help="method for input transformation; one of None, 'log','log10','asinh'")
    tiledbgroup.add_argument("--tdb_indexer",default=None,help="tiledb paths for each input task")
    tiledbgroup.add_argument("--tdb_partition_attribute_for_upsample",default="idr_peak",help="tiledb attribute to use for upsampling, i.e. idr_peak")
    tiledbgroup.add_argument("--tdb_partition_thresh_for_upsample",type=float,default=1,help="values >= partition_thresh_for_upsample within the partition_attribute_for_upsample will be upsampled during training")
    tiledbgroup.add_argument("--upsample_ratio_list_predict",type=float,nargs="*")    
    tiledbgroup.add_argument("--chrom_sizes",default=None,help="chromsizes file for use with tiledb generator")
    tiledbgroup.add_argument("--tiledb_stride",type=int,default=1)

    input_filtering_params=parser.add_argument_group("input_filtering_params")    
    input_filtering_params.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file or hammock) and expanded args.flank to the left and right")
    
    output_params=parser.add_argument_group("output_params")
    output_params.add_argument('--predictions_and_labels_hdf5',help='name of hdf5 to save predictions',default=None)
    output_params.add_argument('--performance_metrics_classification_file',nargs="+", help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    output_params.add_argument('--performance_metrics_regression_file',nargs="+", help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    output_params.add_argument('--performance_metrics_profile_file',nargs="+", help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    
    calibration_params=parser.add_argument_group("calibration_params")
    calibration_params.add_argument("--calibrate_classification",action="store_true",default=False)
    calibration_params.add_argument("--calibrate_regression",action="store_true",default=False)        
    
    model_params=parser.add_argument_group("model_params")
    model_params.add_argument('--weights',help='weights file for the model')
    model_params.add_argument('--yaml',help='yaml file for the model')
    model_params.add_argument('--json',help='json file for the model')
    model_params.add_argument('--functional',default=False,help='use this flag if your model is a functional model',action="store_true")
    model_params.add_argument('--squeeze_input_for_gru',action='store_true')
    model_params.add_argument("--expand_dims",default=False,action="store_true")
    model_params.add_argument("--num_inputs",type=int)
    model_params.add_argument("--num_outputs",type=int)

    arch_params=parser.add_argument_group("arch_params")
    arch_params.add_argument("--from_checkpoint_arch",default=None)
    arch_params.add_argument("--architecture_spec",type=str,default="basset_architecture_multitask")
    arch_params.add_argument("--architecture_from_file",type=str,default=None)
    arch_params.add_argument("--num_tasks",type=int)
    arch_params.add_argument("--tasks",nargs="*",default=None)
    arch_params.add_argument("--task_indices",nargs="*",default=None,help="list of tasks to train on, by index of their position in tdb matrix")
        
    parallelization_params=parser.add_argument_group("parallelization")
    parallelization_params.add_argument("--threads",type=int,default=1)
    parallelization_params.add_argument("--max_queue_size",type=int,default=100)
    parallelization_params.add_argument("--num_gpus",type=int,default=1)

    snp_params=parser.add_argument_group("snp_params")
    snp_params.add_argument("--vcf_file",default=None)
    snp_params.add_argument("--global_vcf",action="store_true")
    snp_params.add_argument('--background_freqs',default=None)
    snp_params.add_argument('--flank',default=500,type=int)
    snp_params.add_argument('--mask',default=10,type=int)
    snp_params.add_argument('--ref_col',type=int,default=None)
    snp_params.add_argument('--alt_col',type=int,default=None)

    train_val_splits=parser.add_argument_group("train_val_splits")
    train_val_splits.add_argument("--num_train",type=int,default=700000)
    train_val_splits.add_argument("--num_valid",type=int,default=150000)

    batch_params=parser.add_argument_group("batch_params")
    batch_params.add_argument("--batch_size",type=int,default=1000)
    batch_params.add_argument("--revcomp",action="store_true")
    batch_params.add_argument("--label_transformer",nargs="+",default=None,help="transformation to apply to label values")
    batch_params.add_argument("--upsample_thresh_list_train",type=float,nargs="*",default=None)
    batch_params.add_argument("--upsample_ratio_list_train",type=float,nargs="*",default=None)
    batch_params.add_argument("--upsample_thresh_list_eval",type=float,nargs="*",default=None)
    batch_params.add_argument("--upsample_ratio_list_eval",type=float,nargs="*",default=None)

    
    weights_params=parser.add_argument_group("weights_params")
    weights_params.add_argument("--init_weights",default=None)
    weights_params.add_argument('--w1',nargs="*", type=float, default=None)
    weights_params.add_argument('--w0',nargs="*", type=float, default=None)
    weights_params.add_argument("--w1_w0_file",default=None)
    weights_params.add_argument("--save_w1_w0", default=None,help="output text file to save w1 and w0 to")
    weights_params.add_argument("--weighted",action="store_true",help="separate task-specific weights denoted with w1, w0 args are to be used")
    weights_params.add_argument("--from_checkpoint_weights",default=None)

    epoch_params=parser.add_argument_group("epoch_params")
    epoch_params.add_argument("--epochs",type=int,default=40)
    epoch_params.add_argument("--patience",type=int,default=3)
    epoch_params.add_argument("--patience_lr",type=int,default=2,help="number of epochs with no drop in validation loss after which to reduce lr")
    epoch_params.add_argument("--shuffle_epoch_start",action='store_true',default=False)
    epoch_params.add_argument("--shuffle_epoch_end",action='store_true',default=False)

    vis_params=parser.add_argument_group("visualization")            
    vis_params.add_argument("--tensorboard",action="store_true")
    vis_params.add_argument("--tensorboard_logdir",default="logs")
    vis_params.add_argument("--trackables",nargs="*",default=['loss','val_loss'], help="list of things to track per batch, such as logcount_predictions_loss,loss,profile_predictions_loss,val_logcount_predictions_loss,val_loss,val_profile_predictions_loss")
    return parser.parse_args()

def cross_validate(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 

    #run training on each of the splits
    if args.assembly not in splits:
        raise Exception("You did not provide the args.assembly flag, or you provided an unsupported genome assembly:"+str(args.assembly)+". Supported assemblies include:"+str(splits.keys())+"; add splits for this assembly to splits.py file")
    args_dict=vars(args)
    print(args_dict) 
    base_model_file=args_dict['model_prefix']
    base_performance_classification_file=args_dict['performance_metrics_classification_file']
    base_performance_regression_file=args_dict['performance_metrics_regression_file']
    base_performance_profile_file=args_dict['performance_metrics_profile_file']

    base_predictions_and_labels_hdf5=args_dict['predictions_and_labels_hdf5']
    base_init_weights=args_dict['init_weights'] 

    all_splits=splits[args.assembly]
    if args.splits!=None:
        all_splits=args.splits 
    for split in all_splits: 
        print("Starting split:"+str(split))

        test_chroms=splits[args.assembly][split]['test']
        validation_chroms=splits[args.assembly][split]['valid']
        train_chroms=list(set(chroms[args.assembly])-set(test_chroms+validation_chroms))

        #convert args to dict
        args_dict=vars(args)
        args_dict['train_chroms']=train_chroms
        args_dict['validation_chroms']=validation_chroms
        
        if base_init_weights is not None:
            args_dict['init_wights']=base_init_weights+"."+str(split)
        #set the training arguments specific to this fold 
        args_dict['model_prefix']=base_model_file+"."+str(split)
        args_dict['tdb_array']=None
        args_dict['load_model_hdf5']=None
        print("Training model on split"+str(split)) 
        train(args_dict)
        
        #set the prediction arguments specific to this fold
        if args.save_w1_w0!=None:
            args_dict["w1_w0_file"]=args.save_w1_w0
        if base_predictions_and_labels_hdf5!=None:
            args_dict['predictions_and_labels_hdf5']=base_predictions_and_labels_hdf5+"."+str(split)
            print(args_dict['predictions_and_labels_hdf5'])
        args_dict['predict_chroms']=test_chroms
        print("Calculating predictions on the test fold in split:"+str(split)) 
        predict(args_dict)

        #score the predictions
        print("scoring split:"+str(split))
        preds_to_score=[]
        labels_to_score=[]
        for cur_output in range(args.num_outputs):
            cur_preds=base_predictions_and_labels_hdf5+"."+str(split)+".predictions."+str(cur_output)
            cur_labels=base_predictions_and_labels_hdf5+"."+str(split)+".labels."+str(cur_output)
            preds_to_score.append(cur_preds)
            labels_to_score.append(cur_labels)
                
        if base_performance_classification_file is not None:
            args_dict['performance_metrics_classification_file']=[perf_file+"."+str(split) for perf_file in base_performance_classification_file]
        elif base_performance_regression_file is not None:
            args_dict['performance_metrics_regression_file']=[perf_file +'.'+str(split) for perf_file in base_performance_regression_file]
        elif base_performance_profile_file is not None:
            args_dict['performance_metrics_profile_file']=[perf_file +'.'+str(split) for perf_file in base_performance_profile_file]
        print(preds_to_score)
        print(labels_to_score)
        args_dict['predictions_hdf5']=preds_to_score
        args_dict['labels_hdf5']=labels_to_score
        get_performance_metrics(args)
        
        
def main():
    args=parse_args()
    cross_validate(args) 

if __name__=="__main__":
    main()
