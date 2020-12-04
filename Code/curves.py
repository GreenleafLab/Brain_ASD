#Draws PRC and ROC curves for predicted and true values

import argparse
from sklearn.metrics import precision_recall_curve,average_precision_score,roc_curve
import matplotlib.pyplot as plt
import h5py
import pickle
import numpy as np
from random import random
from operator import add
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

def parse_args():
    parser=argparse.ArgumentParser("generate PRC & ROC curves")
    parser.add_argument("--truth_hdf5")
    parser.add_argument("--prediction_pickle")
    parser.add_argument("--out_prefix")
    parser.add_argument("--labels")
    parser.add_argument("--title") 
    return parser.parse_args()

def filter_vals(precision,recall):
    thresholds=[i/100.0 for i in range(100,0,-1)]
    recall, precision = zip(*sorted(zip(recall, precision),reverse=True))
    recall=list(recall)
    precision=list(precision)
    rounded_recall=[round(i,2) for i in recall]
    filtered_precision=[]
    filtered_recall=[]
    cur_precision=0 
    for t in thresholds:
        if t in rounded_recall:            
            cur_index=rounded_recall.index(t)
            cur_precision=precision[cur_index]   
        filtered_precision.append(cur_precision)
        filtered_recall.append(t)
    return filtered_precision,filtered_recall

def main():
    args=parse_args()
    with open(args.prediction_pickle,'rb') as handle:
        y_pred=pickle.load(handle)
    y_true=h5py.File(args.truth_hdf5,'r')
    y_true=np.asarray(y_true['Y']['default_output_mode_name'])
    labels=open(args.labels,'r').read().strip().split('\n')[0].split('\t')[1::]
    mean_precision=[]
    mean_recall=[]
    mean_fpr=[]
    mean_tpr=[]

    #initialize figure
    fig1=plt.figure()
    ax1=fig1.add_subplot(121)
    ax2=fig1.add_subplot(122)
    #compute precision & recall for each task 
    for i in range(len(labels)):
        
        precision, recall, thresholds = precision_recall_curve(y_true[:,i], y_pred[:,i])
        fpr,tpr,thresholds=roc_curve(y_true[:,i],y_pred[:,i])
        
        cur_color=([random(),random(),random()])
        ax1.step(recall, precision, color=cur_color, alpha=0.2,where='post',label=labels[i])
        ax2.step(fpr,tpr,color=cur_color,alpha=0.2,where='post',label=labels[i])

        precision,recall=filter_vals(precision,recall)
        tpr,fpr=filter_vals(tpr,fpr)

        if len(mean_precision)==0:
            mean_precision=precision
        else:
            mean_precision=map(add, mean_precision,precision)
        if len(mean_recall)==0:
            mean_recall=recall
        else:
            mean_recall=map(add,mean_recall,recall)
        if len(mean_tpr)==0:
            mean_tpr=tpr
        else:
            mean_tpr=map(add,mean_tpr,tpr)
        if len(mean_fpr)==0:
            mean_fpr=fpr
        else:
            mean_fpr=map(add,mean_fpr,fpr)

    #get average metric values 
    mean_precision=[i/len(labels) for i in mean_precision]
    mean_recall=[i/len(labels) for i in mean_recall]
    mean_tpr=[i/len(labels) for i in mean_tpr]
    mean_fpr=[i/len(labels) for i in mean_fpr]
    
    ax1.step(mean_recall, mean_precision, color=(0,0,0), linewidth=2,
             where='post',label="mean")
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlim([0.0, 1.0])
    ax1.set_title(args.title)

    ax2.step(mean_fpr,mean_tpr,color=(0,0,0),linewidth=2,
             where='post',label='mean')
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_ylim([0.0,1.05])
    ax2.set_xlim([0.0,1.0])
    ax2.set_title(args.title)

    #generate output file
    outf=open(args.out_prefix+'.average.metrics','w')
    outf.write('Precision\tRecall\tFPR\tTPR\n')
    for i in range(len(mean_precision)):
        outf.write(str(mean_precision[i])+'\t'+str(mean_recall[i])+'\t'+str(mean_fpr[i])+'\t'+str(mean_tpr[i])+'\n')
    #plt.savefig(args.out_prefix+".png",dpi=100)
    plt.show()
if __name__=="__main__":
    main()
    
