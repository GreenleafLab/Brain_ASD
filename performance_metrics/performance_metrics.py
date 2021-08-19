from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import warnings
import pandas as pd
from .classification_performance_metrics import *
from .regression_performance_metrics import *
from .bpnet_performance_metrics import *

def parse_args():
    parser=argparse.ArgumentParser(description='Provide a model prediction pickle to compute performance metrics.')
    parser.add_argument('--sample_N',type=int,default=None,help="sample N coordinates at random for scoring")
    parser.add_argument('--chunk_size',type=int,default=None,help="Number of lines to load at once")
    parser.add_argument('--labels_hdf5',nargs="+",default=None)
    parser.add_argument('--predictions_hdf5',nargs="+",default=None)
    parser.add_argument('--predictions_pickle_to_load',help="if predictions have already been generated, provide a pickle with them to just compute the accuracy metrics",default=None)
    parser.add_argument('--performance_metrics_classification_file',nargs="+",help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--performance_metrics_regression_file',nargs="+",help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--performance_metrics_profile_file',nargs="+",help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--tasks',nargs="+",default=None,help="comma-separated list of task names for each dataset to score")
    parser.add_argument('--task_indices',nargs="+",type=int,default=None) 
    return parser.parse_args()

def get_metrics_function(args):
    if args.performance_metrics_classification_file is not None:
        return get_performance_metrics_classification
    elif args.performance_metrics_regression_file is not None:
        return get_performance_metrics_regression
    elif args.performance_metrics_profile_file is not None:
        return get_performance_metrics_profile
    else:
        raise Exception("one of --performance_metrics_classification_file, --performance_metrics_regression_file, --performance_metrics_profile_file must be provided")

def get_output_file(args,i):
    if args.performance_metrics_classification_file is not None:
        return args.performance_metrics_classification_file[i]
    elif args.performance_metrics_regression_file is not None:
        return args.performance_metrics_regression_file[i]
    elif args.performance_metrics_profile_file is not None:
        return args.performance_metric_profile_file[i]
    else:
        raise Exception("one of --performance_metrics_classification_file, --performance_metrics_regression_file, --performance_metrics_profile_file must be provided")
    
def metrics_from_pickle(cur_pickle,tasks,args):
    with open(cur_pickle,'rb') as handle:
        predictions=pickle.load(handle)
        labels=predictions['labels'][tasks]
        model_predictions=predictions['predictions'][tasks]
        if 'calibrated_predictions' in predictions.keys():
            model_predictions=predictions['calibrated_predictions'][tasks]
        metrics_function=get_metrics_function(args)
        return metrics_function(model_predictions,labels)

def metrics_from_hdf(cur_labels, cur_predictions, tasks,args):
    cur_labels=pd.read_hdf(cur_labels)[tasks]
    cur_predictions=pd.read_hdf(cur_predictions)[tasks]
    #make sure they are ordered
    cur_labels=cur_labels.loc[cur_predictions.index]
    metrics_function=get_metrics_function(args)
    return metrics_function(cur_predictions,cur_labels) 

def get_performance_metrics(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args)

    #get metrics from hdf5
    if args.labels_hdf5 is not None:
        assert args.predictions_hdf5 is not None
        num_datasets=len(args.labels_hdf5)
        for i in range(num_datasets):
            cur_labels=args.labels_hdf5[i]
            cur_predictions=args.predictions_hdf5[i]
            if args.task_indices is not None:
                cur_tasks=args.task_indices[i]
            elif args.tasks is not None:
                cur_tasks=args.tasks[i].split(',')
            else: 
                cur_tasks=[j for j in range(cur_predictions.shape[1])]
            if type(cur_tasks) is not list:
                cur_tasks=[cur_tasks]
            cur_metrics=metrics_from_hdf(cur_labels,cur_predictions,cur_tasks,args)
            outfile=get_output_file(args,i)
            write_performance_metrics(outfile,cur_metrics,cur_tasks)

    #get metrics from pickle
    elif args.predictions_pickle_to_load is not None:
        num_datasets=len(args.predictions_pickle_to_load)
        for i in range(num_datasets):
            cur_pickle=args.predictions_pickle_to_load[i]
            if args.task_indices is not None:
                cur_tasks=args.task_indices[i]
            elif args.tasks is not None:
                cur_tasks=args.tasks[i].split(',')
            else: 
                cur_tasks=[j for j in range(cur_predictions.shape[1])]                
            if type(cur_tasks) is not list:
                cur_tasks=[cur_tasks]
            cur_metrics=metrics_from_pickle(cur_pickle,cur_tasks,args)
            outfile=get_output_file(args,i)
            write_performance_metrics(outfile,cur_metrics,cur_tasks)
    
#write performance metrics to output file: 
def write_performance_metrics(output_file,metrics_dict,tasks):
    print(metrics_dict) 
    metrics_df=pd.DataFrame(metrics_dict,index=tasks).transpose()
    metrics_df.to_csv(output_file,sep='\t',header=True,index=True)
    print(metrics_df)

def main():
    args=parse_args()
    get_performance_metrics(args)
    
if __name__=="__main__":
    main()
    
