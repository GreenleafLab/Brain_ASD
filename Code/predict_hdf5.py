from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
import time
#graceful shutdown
import psutil
import signal 
import os

#multithreading
#from concurrent.futures import ProcessPoolExecutor, as_completed
#from multiprocessing import Pool,Process, Queue 

import warnings
import numpy as np
import pysam
import pandas as pd

import tensorflow as tf 
from kerasAC.activations import softMaxAxis1
from .calibrate import * 
from .generators.basic_generator import *
from .generators.tiledb_predict_generator import *
from .tiledb_config import *
from .s3_sync import *
from .splits import *
from .get_model import *
from .custom_losses import * 
from kerasAC.config import args_object_from_args_dict
from kerasAC.performance_metrics import *
from kerasAC.custom_losses import *
from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1
import argparse
import yaml 
import h5py 
import pickle
import numpy as np 
import tensorflow.keras as keras 
from keras.losses import *
from keras.models import Model
from keras.utils import multi_gpu_model
from kerasAC.custom_losses import *
from abstention.calibration import PlattScaling, IsotonicRegression 
import random
import pdb 

def parse_args():
    parser=argparse.ArgumentParser(description='Provide model files  & a dataset, get model predictions')
    input_data_path=parser.add_argument_group('input_data_path')
    input_data_path.add_argument("--index_data_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels")
    input_data_path.add_argument("--index_tasks",nargs="*",default=None)    
    input_data_path.add_argument("--input_data_path",nargs="+",default=None,help="seq or path to seqdataloader hdf5")
    input_data_path.add_argument("--output_data_path",nargs="+",default=None,help="path to seqdataloader hdf5")
    
    input_data_path.add_argument('--variant_bed',default=None)
    input_data_path.add_argument('--ref_fasta')

    input_filtering_params=parser.add_argument_group("input_filtering_params")    
    input_filtering_params.add_argument('--predict_chroms',nargs="*",default=None)
    input_filtering_params.add_argument("--genome",default=None)
    input_filtering_params.add_argument("--fold",type=int,default=None)
    input_filtering_params.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file or hammock) and expanded args.flank to the left and right")
    input_filtering_params.add_argument("--tasks",nargs="*",default=None)
    
    output_params=parser.add_argument_group("output_params")
    output_params.add_argument('--predictions_and_labels_hdf5',help='name of hdf5 to save predictions',default=None)
    calibration_params=parser.add_argument_group("calibration_params")
    calibration_params.add_argument("--calibrate_classification",action="store_true",default=False)
    calibration_params.add_argument("--calibrate_regression",action="store_true",default=False)        
    
    weight_params=parser.add_argument_group("weight_params")
    weight_params.add_argument('--w1',nargs="*",type=float)
    weight_params.add_argument('--w0',nargs="*",type=float)
    weight_params.add_argument("--w1_w0_file",default=None)


    model_params=parser.add_argument_group("model_params")
    model_params.add_argument('--load_model_hdf5',help='hdf5 file that stores the model')
    model_params.add_argument('--weights',help='weights file for the model')
    model_params.add_argument('--yaml',help='yaml file for the model')
    model_params.add_argument('--json',help='json file for the model')
    model_params.add_argument('--functional',default=False,help='use this flag if your model is a functional model',action="store_true")
    model_params.add_argument('--squeeze_input_for_gru',action='store_true')
    model_params.add_argument("--expand_dims",default=False,action='store_true')
    model_params.add_argument("--num_inputs",type=int)
    model_params.add_argument("--num_outputs",type=int)
    model_params.add_argument("--num_gpus",type=int,default=1)
    
    parallelization_params=parser.add_argument_group("parallelization")
    parallelization_params.add_argument("--threads",type=int,default=1)
    parallelization_params.add_argument("--max_queue_size",type=int,default=100)

    snp_params=parser.add_argument_group("snp_params")
    snp_params.add_argument('--background_freqs',default=None)
    snp_params.add_argument('--flank',default=500,type=int)
    snp_params.add_argument('--mask',default=10,type=int)
    snp_params.add_argument('--ref_col',type=int,default=None)
    snp_params.add_argument('--alt_col',type=int,default=None)

    parser.add_argument('--batch_size',type=int,help='batch size to use to make model predictions',default=50)
    return parser.parse_args()

def get_out_predictions_prefix(args):
    if args.predictions_and_labels_hdf5.startswith('s3://'):
        #use a local version of the file and upload to s3 when finished
        bucket,filename=s3_string_parse(args.predictions_and_labels_hdf5)
        out_predictions_prefix=filename.split('/')[-1]+".predictions"
    else: 
        out_predictions_prefix=args.predictions_and_labels_hdf5+".predictions"
    if args.calibrate_classification is True:
        out_predictions_prefix=out_predictions_prefix+".logits"
    elif args.calibrate_regression is True:
        out_predictions_prefix=out_predictions_prefix+".preacts"
    return out_predictions_prefix 

def get_out_calibrated_prefix(args):
    if args.predictions_and_labels_hdf5.startswith('s3://'):
        #use a local version of the file and upload to s3 when finished
        bucket,filename=s3_string_parse(args.predictions_and_labels_hdf5)
        out_predictions_prefix=filename.split('/')[-1]+".predictions.calibrated"
    else: 
        out_predictions_prefix=args.predictions_and_labels_hdf5+".predictions.calibrated"
    return out_predictions_prefix 
    

def get_out_labels_prefix(args):
    if args.predictions_and_labels_hdf5.startswith('s3://'):
        #use a local version of the file and upload to s3 when finished
        bucket,filename=s3_string_parse(args.predictions_and_labels_hdf5)
        out_labels_prefix=filename.split('/')[-1]+".labels"
    else: 
        out_labels_prefix=args.predictions_and_labels_hdf5+".labels" 
    return out_labels_prefix

def write_predictions(args):
    '''
    separate predictions file for each output/task combination 
    '''
    try:
        out_predictions_prefix=get_out_predictions_prefix(args)
        first=True
        while True:
            pred_df=pred_queue.get()
            if type(pred_df) == str: 
                if pred_df=="FINISHED":
                    return
            if first is True:
                mode='w'
                first=False
                append=False
            else:
                mode='a'
                append=True
            for cur_output_index in range(len(pred_df)):
                #get cur_pred_df for current output
                cur_pred_df=pred_df[cur_output_index]
                if args.tasks is not None:
                    task_names={}
                    for i in range(len(args.tasks)):
                        task_names[i]=args.tasks[i]
                    cur_pred_df=cur_pred_df.rename(columns=task_names)
                cur_out_f='.'.join([out_predictions_prefix,str(cur_output_index)])
                cur_pred_df.to_hdf(cur_out_f,key="data",mode=mode,append=append,format="table",min_itemsize={'CHR':30})
                
    except KeyboardInterrupt:
        #shutdown the pool
        # Kill remaining child processes
        kill_child_processes(os.getpid())
        raise 
    except Exception as e:
        print(e)
        #shutdown the pool
        # Kill remaining child processes
        kill_child_processes(os.getpid())
        raise e

def write_labels(args):
    '''
    separate label file for each output/task combination
    '''
    out_labels_prefix=get_out_labels_prefix(args)
    try:
        first=True
        while True:
            label_df=label_queue.get()
            if type(label_df)==str:
                if label_df=="FINISHED":
                    return
            if first is True:
                mode='w'
                first=False
                append=False
            else:
                mode='a'
                append=True
            for cur_output_index in range(len(label_df)):
                cur_label_df=label_df[cur_output_index]
                if args.tasks is not None:
                    task_names={}
                    for i in range(len(args.tasks)):
                        task_names[i]=args.tasks[i]
                    cur_label_df=cur_label_df.rename(columns=task_names)

                cur_out_f='.'.join([out_labels_prefix,str(cur_output_index)])
                cur_label_df.to_hdf(cur_out_f,key="data",mode=mode,append=append,format="table",min_itemsize={'CHR':30})
                
    except KeyboardInterrupt:
        #shutdown the pool
        # Kill remaining child processes
        kill_child_processes(os.getpid())
        raise 
    except Exception as e:
        print(e)
        #shutdown the pool
        # Kill remaining child processes
        kill_child_processes(os.getpid())
        raise e
    return

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = mp.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)
        


def get_batch_wrapper(idx):
    X,y,coords=test_generator[idx]
    if type(y) is not list:
        y=[y]
    try:
        y=[i.squeeze(axis=-1) for i in y]
    except:
        pass
    if type(X) is not list:
        X=[X]
    
    #represent coords w/ string, MultiIndex not supported in table append mode
    #set the column names for the MultiIndex
    coords=pd.MultiIndex.from_tuples(coords,names=['CHR','START','END'])
    y=[pd.DataFrame(i,index=coords) for i in y]
    return [X,y,coords,idx]


def get_hdf5_predict_generator(args):
    global test_generator
    test_chroms=get_chroms(args,split='test')
    test_generator=DataGenerator(index_path=args.index_data_path,
                                 input_path=args.input_data_path,
                                 output_path=args.output_data_path,
                                 index_tasks=args.index_tasks,
                                 num_inputs=args.num_inputs,
                                 num_outputs=args.num_outputs,
                                 ref_fasta=args.ref_fasta,
                                 batch_size=args.batch_size,
                                 add_revcomp=False,
                                 chroms_to_use=test_chroms,
                                 expand_dims=args.expand_dims,
                                 tasks=args.tasks,
                                 shuffle=False,
                                 return_coords=True)
    return test_generator
def get_variant_predict_generator(args):
    global test_generator
    test_chroms=get_chroms(args,split='test')
    test_generator=SNPGenerator(args.allele_col,
                                args.flank,
                                index_path=args.index_data_path,
                                input_path=args.input_data_path,
                                output_path=args.output_data_path,
                                index_tasks=args.index_tasks,
                                num_inputs=args.num_inputs,
                                num_outputs=args.num_outputs,
                                ref_fasta=args.ref_fasta,
                                allele_col=args.ref_col,
                                batch_size=args.batch_size,
                                add_revcomp=False,
                                chroms_to_use=test_chroms,
                                expand_dims=args.expand_dims,
                                tasks=args.tasks,
                                shuffle=False,
                                return_coords=True)
    return test_generator

def get_generator(args):
    if args.variant_bed is not None:
        return get_variant_predict_generator(args)
    else:
        return get_hdf5_predict_generator(args)

def predict_on_batch_wrapper(args,model,test_generator):
    num_batches=len(test_generator)
    processed=0
    try:
        with mp.Pool(processes=args.threads,initializer=init_worker) as pool: 
            while (processed < num_batches):
                idset=range(processed,min([num_batches,processed+args.max_queue_size]))
                for result in pool.imap_unordered(get_batch_wrapper,idset):
                    X=result[0]
                    y=result[1]
                    coords=result[2]
                    idx=result[3]
                    processed+=1
                    if processed%10==0:
                        print(str(processed)+"/"+str(num_batches))
                    #get the model predictions            
                    preds=model.predict_on_batch(X)
                    if type(preds) is not list:
                        preds=[preds]
                    try:
                        preds=[i.squeeze(axis=-1) for i in preds]
                    except:
                        pass 
                    preds_dfs=[pd.DataFrame(cur_pred,index=coords) for cur_pred in preds]
                    label_queue.put(y)
                    pred_queue.put(preds_dfs)
                    
    except KeyboardInterrupt:
        #shutdown the pool
        pool.terminate()
        pool.join() 
        # Kill remaining child processes
        kill_child_processes(os.getpid())
        raise 
    except Exception as e:
        print(e)
        #shutdown the pool
        pool.terminate()
        pool.join()
        # Kill remaining child processes
        kill_child_processes(os.getpid())
        raise e
    print("finished with tiledb predictions!")
    label_queue.put("FINISHED")
    label_queue.close() 
    pred_queue.put("FINISHED")
    pred_queue.close() 
    return


def get_model_layer_functor(model,target_layer_idx):
    from keras import backend as K
    inp=model.input
    outputs=model.layers[target_layer_idx].output
    functor=K.function([inp], [outputs])
    return functor 

def get_layer_outputs(functor,X):
    return functor([X])

def predict(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 
    global pred_queue
    global label_queue
    
    pred_queue=mp.Queue()
    label_queue=mp.Queue()
    
    label_writer=mp.Process(target=write_predictions,args=([args]))
    pred_writer=mp.Process(target=write_labels,args=([args]))
    label_writer.start()
    pred_writer.start() 


    #get the generator
    test_generator=get_generator(args) 
    
    #get the model
    #if calibration is to be done, get the preactivation model 
    model=get_model(args)
    perform_calibration=args.calibrate_classification or args.calibrate_regression
    if perform_calibration==True:
        if args.calibrate_classification==True:
            print("getting logits")
            model=Model(inputs=model.input,
                               outputs=model.layers[-2].output)
        elif args.calibrate_regression==True:
            print("getting pre-relu outputs (preacts)")
            model=Model(inputs=model.input,
                        outputs=model.layers[-1].output)
            
    #call the predict_on_batch_wrapper
    predict_on_batch_wrapper(args,model,test_generator)

    #drain the queue
    try:
        while not label_queue.empty():
            print("draining the label Queue")
            time.sleep(2)
    except Exception as e:
        print(e)
    try:
        while not pred_queue.empty():
            print("draining the prediction Queue")
            time.sleep(2)
    except Exception as e:
        print(e)
    
    print("joining label writer") 
    label_writer.join()
    print("joining prediction writer") 
    pred_writer.join()

    #sync files to s3 if needed
    if args.predictions_and_labels_hdf5.startswith('s3://'):
        #use a local version of the file and upload to s3 when finished
        bucket,filename=s3_string_parse(args.predictions_and_labels_hdf5)
        out_predictions_prefix=filename.split('/')[-1]+".predictions"
        out_labels_prefix=filename.split('/')[-1]+".predictions"
        #upload outputs for all tasks
        import glob
        to_upload=[]
        for f in glob.glob(out_predictions_prefix+"*"):
            s3_path='/'.join(args.predictions_and_labels_hdf5.split('/')[0:-1])+"/"+f
            upload_s3_file(s3_path,f)
        for f in glob.glob(out_labels_prefix+"*"):
            s3_path='/'.join(args.predictions_and_labels_hdf5.split('/')[0:-1])+"/"+f
            upload_s3_file(s3_path,f)
        
    #perform calibration, if specified
    if perform_calibration is True:
        print("calibrating")
        out_predictions_prefix=get_out_predictions_prefix(args)
        out_labels_prefix=get_out_labels_prefix(args)
        out_calibrated_prefix=get_out_calibrated_prefix(args)
        
        for output_index in range(args.num_outputs):
            preacts_fname='.'.join([out_predictions_prefix,str(output_index)])
            labels_fname='.'.join([out_labels_prefix,str(output_index)])
            calibrated_fname='.'.join([out_calibrated_prefix,str(output_index)])
            #model has already been transformed to a preact model above
            calibrate(preacts_fname,
                      labels_fname,
                      model,
                      calibrated_fname,
                      calibrate_regression=args.calibrate_regression,
                      calibrate_classification=args.calibrate_classification,
                      get_model_preacts=False)
    #clean up any s3 artifacts:
    run_cleanup()
    
def main():
    args=parse_args()
    predict(args)


if __name__=="__main__":
    main()
    
