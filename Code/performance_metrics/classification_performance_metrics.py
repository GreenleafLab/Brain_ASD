from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import warnings
import numpy as np
import pysam
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve 
from collections import OrderedDict, defaultdict
from .utils import * 


def auroc_func(predictions_for_task_filtered, true_y_for_task_filtered):
    try:
        true_y_for_task_filtered=[int(round(i)) for i in true_y_for_task_filtered]
        task_auroc = roc_auc_score(y_true=true_y_for_task_filtered,
                                   y_score=predictions_for_task_filtered)
    except Exception as e:
        #if there is only one class in the batch of true_y, then auROC cannot be calculated
        print("Could not calculate auROC:")
        print(str(e))
        task_auroc=None 
    return task_auroc

def auprc_func(predictions_for_task_filtered, true_y_for_task_filtered):
    # sklearn only supports 2 classes (0,1) for the auPRC calculation
    try:
        true_y_for_task_filtered=[int(round(i)) for i in true_y_for_task_filtered]
        task_auprc=average_precision_score(true_y_for_task_filtered, predictions_for_task_filtered)
    except:
        print("Could not calculate auPRC:")
        print(sys.exc_info()[0])
        task_auprc=None 
    return task_auprc

def get_accuracy_stats_for_task(predictions_for_task_filtered, true_y_for_task_filtered, c):
    predictions_for_task_filtered_round = np.array([round(el) for el in predictions_for_task_filtered])
    accuratePredictions = predictions_for_task_filtered_round==true_y_for_task_filtered;

    numPositives_forTask=np.sum(true_y_for_task_filtered>0,axis=0,dtype="float");
    numNegatives_forTask=np.sum(true_y_for_task_filtered<=0,axis=0,dtype="float"); 

    accuratePredictions_positives = np.sum(accuratePredictions*(true_y_for_task_filtered>0),axis=0);
    accuratePredictions_negatives = np.sum(accuratePredictions*(true_y_for_task_filtered<=0),axis=0);
    unbalancedAccuracy_forTask = (accuratePredictions_positives + accuratePredictions_negatives)/(numPositives_forTask + numNegatives_forTask)

    positiveAccuracy_forTask = accuratePredictions_positives/numPositives_forTask
    negativeAccuracy_forTask = accuratePredictions_negatives/numNegatives_forTask
    balancedAccuracy_forTask= (positiveAccuracy_forTask+negativeAccuracy_forTask)/2;
    returnDict={'unbalanced_accuracy':unbalancedAccuracy_forTask,
                'positive_accuracy':positiveAccuracy_forTask,
                'negative_accuracy':negativeAccuracy_forTask,
                'balanced_accuracy':balancedAccuracy_forTask,
                'num_positives':numPositives_forTask,
                'num_negatives':numNegatives_forTask}
    return returnDict


def recall_at_fdr_function(predictions_for_task_filtered,true_y_for_task_filtered,fdr_thresh_list):
    for fdr_thresh_index in range(len(fdr_thresh_list)):
        if float(fdr_thresh_list[fdr_thresh_index])>1:
            fdr_thresh_list[fdr_thresh_index]=fdr_thresh_list[fdr_thresh_index]/100
    
    precision,recall,class_thresholds=precision_recall_curve(true_y_for_task_filtered,predictions_for_task_filtered)
    fdr=1-precision

    #remove the last values in recall and fdr, as the scipy precision_recall_curve function sets them to 0 automatically
    recall=np.delete(recall,-1)
    fdr=np.delete(fdr,-1)
    df=pd.DataFrame({'recall':recall,'fdr':fdr,'class_thresholds':class_thresholds})
    df=df.sort_values(by=['fdr','recall','class_thresholds'])

    #get the recall, fdr at each thresh
    recall_thresholds=[]
    class_thresholds=[]
    for fdr_thresh in fdr_thresh_list:
        try:
            recall_thresholds.append(float(df[df.fdr<=fdr_thresh].tail(n=1)['recall']))
            class_thresholds.append(float(df[df.fdr<=fdr_thresh].tail(n=1)['class_thresholds']))
        except:
            print("No class threshold can give requested fdr <=:"+str(fdr_thresh))
            recall_thresholds.append(np.nan)
            class_thresholds.append(np.nan)
            
    return recall_thresholds, class_thresholds

def get_performance_metrics_classification(predictions,true_y):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    #make sure the chromosome regions are sorted in the same order in the prediction file and the label file
    if type(predictions)==pd.DataFrame:
        assert sum(predictions.index!=true_y.index)==0;
    [num_rows, num_cols]=true_y.shape
    if type(predictions)==pd.DataFrame:
        predictions=predictions.values
        true_y=true_y.values
        
    performance_stats=None
    for c in range(num_cols):
        true_y_for_task=np.squeeze(true_y[:,c])
        predictions_for_task=np.squeeze(predictions[:,c])
        predictions_for_task_filtered,true_y_for_task_filtered = remove_ambiguous_peaks(predictions_for_task,true_y_for_task)
        print("predictions:"+str(predictions_for_task_filtered))
        print("labels:"+str(true_y_for_task_filtered))
        print(c) 
        accuracy_stats_task = get_accuracy_stats_for_task(predictions_for_task_filtered, true_y_for_task_filtered, c)
        auprc_task=auprc_func(predictions_for_task_filtered,true_y_for_task_filtered)
        auroc_task=auroc_func(predictions_for_task_filtered,true_y_for_task_filtered)
        recall,class_thresh=recall_at_fdr_function(predictions_for_task_filtered,true_y_for_task_filtered,[50,20,10])
                
        if performance_stats==None:
            performance_stats=dict()
            for key in accuracy_stats_task:
                performance_stats[key]=[accuracy_stats_task[key]]
            performance_stats['auprc']=[auprc_task]
            performance_stats['auroc']=[auroc_task]
            performance_stats['recall_at_fdr_50']=[recall[0]]
            performance_stats['recall_at_fdr_20']=[recall[1]]
            performance_stats['recall_at_fdr_10']=[recall[2]]            
        else:
            for key in accuracy_stats_task:
                performance_stats[key].append(accuracy_stats_task[key])
            performance_stats['auprc'].append(auprc_task)
            performance_stats['auroc'].append(auroc_task)
            performance_stats['recall_at_fdr_50'].append(recall[0])
            performance_stats['recall_at_fdr_20'].append(recall[1])
            performance_stats['recall_at_fdr_10'].append(recall[2])
    print(str(performance_stats))
    return performance_stats  

    
