import pdb
import pandas as pd
import numpy as np 
import argparse
import pyBigWig
from ..helpers.mnnll import * 
#from .utils import *
from scipy.stats import spearmanr, pearsonr
from scipy import nanmean, nanstd
from scipy.special import softmax
from scipy.spatial.distance import jensenshannon
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

plt.rcParams["figure.figsize"]=10,5
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)
#Performance metrics for profile models

def parse_args():
    parser=argparse.ArgumentParser(description="performance metrics for profile models")
    parser.add_argument("--labels",help="each label hdf5 contains labels for a given task in the model")
    parser.add_argument("--predictions",help="each prediction hdf5 contains predictions for a given task from the model")
    parser.add_argument("--losses",nargs="+",help="counts or profile")
    parser.add_argument("--loss_suffixes",nargs="+")
    parser.add_argument("--outf")
    parser.add_argument("--title") 
    parser.add_argument("--pseudoreps",nargs="+",default=None,help="bigwig replicates for calculating upper bound of performance")
    parser.add_argument("--flank",type=int,default=500)
    parser.add_argument("--label_min_to_score",type=float,default=None)
    parser.add_argument("--label_max_to_score",type=float,default=None)
    parser.add_argument("--smooth_observed_profile",action="store_true",default=False)
    parser.add_argument("--smooth_predicted_profile",action="store_true",default=False)
    parser.add_argument("--smooth_preps",action="store_true",default=False)
    return parser.parse_args() 


def density_scatter(x, y, xlab, ylab, title,figtitle, ax = None, sort = True, bins = 20):
    """
    Scatter plot colored by 2d histogram
    """
    print(x.shape)
    print(y.shape)
    
    bad_x=np.where(np.isnan(x))
    bad_y=np.where(np.isnan(y))
    x=x[~np.isin(np.arange(x.size),bad_x)]
    y=y[~np.isin(np.arange(y.size),bad_x)]
    x=x[~np.isin(np.arange(x.size),bad_y)]
    y=y[~np.isin(np.arange(y.size),bad_y)]
    
    print(x.shape)
    print(y.shape) 
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc='best')
    plt.xlim(-7,11)
    plt.ylim(-7,11)
    plt.savefig(figtitle,format='png',dpi=300)    
    return ax



def get_pseudorep_counts_cor(pseudoreps,coords,title,outf,flank=500):
    prep1_vals=[]
    prep2_vals=[] 
    for coord in coords:
        chrom=coord[0]
        center=coord[1]
        start=center-flank
        end=center+flank
        prep1_vals.append(np.log(np.sum(np.nan_to_num(pseudoreps[0].values(chrom,start,end)))+1))
        prep2_vals.append(np.log(np.sum(np.nan_to_num(pseudoreps[1].values(chrom,start,end)))+1))
    spearman_cor=spearmanr(prep1_vals,prep2_vals)[0]
    pearson_cor=pearsonr(prep1_vals,prep2_vals)[0]
    density_scatter(np.asarray(prep1_vals), np.asarray(prep2_vals) ,xlab='Log Count Labels Pseudorep1',ylab='Log Count Labels Pseudorep 2',title="counts:"+str(title)+" spearman R="+str(round(spearman_cor,3))+", Pearson R="+str(round(pearson_cor,3)),figtitle=outf+".counts.pseudorep.png")
    return spearman_cor, pearson_cor
    
def counts_metrics(labels,preds,outf,title,pseudoreps,flank):            
    spearman_cor=spearmanr(labels[0].values,preds[0].values)[0]
    pearson_cor=pearsonr(labels[0].values,preds[0].values)[0]
    mse=((labels[0].values - preds[0].values)**2).mean(axis=0)
    
    
    plt.rcParams["figure.figsize"]=8,8
    plt.figure()
    density_scatter(labels[0].values,
                    preds[0].values,
                    xlab='Log Count Labels',
                    ylab='Log Count Predictions',
                    title="counts:"+title+" spearman R="+str(round(spearman_cor,3))+", Pearson R="+str(round(pearson_cor,3)),
                    figtitle=outf+".counts.png")
    if pseudoreps is not None:
        spearman_cor_ps,pearson_cor_ps=get_pseudorep_counts_cor(pseudoreps,labels.index,title,outf,flank)
    else:
        spearman_cor_ps=None
        pearson_cor_ps=None
    return spearman_cor, pearson_cor, mse, spearman_cor_ps, pearson_cor_ps

def profile_metrics(profile_labels,profile_preds,counts_labels,counts_preds,outf_prefix,title,pseudoreps,flank,smooth_observed_profile, smooth_predicted_profile,smooth_preps):
    #profile-preds is in logit space
    #get the softmax to put in probability space
    coords=profile_labels.index.tolist()
    chroms=[i[0] for i in coords]
    summits=[i[1] for i in coords]
    profile_labels=profile_labels.values
    profile_preds=profile_preds.values

    #perform smoothing of labels/predictions, if specified 
    if smooth_observed_profile==True:
        profile_labels=scipy.ndimage.gaussian_filter1d(profile_labels, 7,axis=1, truncate=(80 / 14))
    if smooth_predicted_profile==True:
        profile_preds=scipy.ndimage.gaussian_filter1d(profile_preds, 7, axis=1, truncate=(80/14))
    
    profile_preds_softmax=softmax(profile_preds,axis=1)

    #get multinomial nll
    print(profile_labels.shape)
    print(profile_preds.shape)
    print(counts_labels.shape)
    
    mnnll_vals=profile_multinomial_nll(np.expand_dims(np.expand_dims(profile_labels,axis=1),axis=-1),
                                       np.expand_dims(np.expand_dims(np.log(profile_preds_softmax),axis=1),axis=-1),
                                       np.expand_dims(np.exp(counts_labels),axis=-1))
    #put the counts in probability space to use jsd
    num_regions=profile_labels.shape[0]
    region_jsd=[]
    pseudorep_jsd=[]
    shuffled_labels_jsd=[] #shuffled labels vs observed labels

    outf=open(outf_prefix+".jsd.txt",'w')
    outf.write('Region\tJSD\tPseudorepJSD\tNLL\n')
    for i in range(num_regions):
        denominator=np.nansum(profile_labels[i,:])
        if denominator!=0:
            cur_profile_labels_prob=profile_labels[i,:]/denominator
        else:
            cur_profile_labels_prob=profile_labels[i,:]
        cur_profile_preds_softmax=profile_preds_softmax[i,:]
        cur_jsd=jensenshannon(cur_profile_labels_prob,cur_profile_preds_softmax)
        region_jsd.append(cur_jsd)
        #get the jsd of shuffled label with true label 
        shuffled_labels=np.random.permutation(profile_labels[i,:])
        shuffled_labels_prob=shuffled_labels/np.nansum(shuffled_labels)
        shuffled_labels_jsd.append(jensenshannon(cur_profile_labels_prob,shuffled_labels_prob))

        if pseudoreps is not None:
            prep1_vals=np.nan_to_num(pseudoreps[0].values(chroms[i],summits[i]-flank,summits[i]+flank,numpy=True))
            prep2_vals=np.nan_to_num(pseudoreps[1].values(chroms[i],summits[i]-flank,summits[i]+flank,numpy=True))
            if smooth_preps==True:
                prep1_vals=scipy.ndimage.gaussian_filter1d(prep1_vals, 7, truncate=(80 / 14))
                prep2_vals=scipy.ndimage.gaussian_filter1d(prep2_vals, 7, truncate=(80 / 14))
            #normalize
            if np.nansum(prep1_vals)!=0:
                prep1_vals=prep1_vals/np.nansum(prep1_vals)
            if np.nansum(prep2_vals)!=0:
                prep2_vals=prep2_vals/np.nansum(prep2_vals)
            prep_jsd=jensenshannon(prep1_vals,prep2_vals)
            pseudorep_jsd.append(prep_jsd)
        else:
            prep_jsd=None
        outf.write(str(chroms[i])+'\t'+str(summits[i])+'\t'+str(cur_jsd)+'\t'+str(prep_jsd)+'\t'+str(mnnll_vals[i])+'\n')
    outf.close() 

    num_bins=100
    plt.rcParams["figure.figsize"]=8,8

    #plot mnnll histogram 
    plt.figure()
    n,bins,patches=plt.hist(mnnll_vals,num_bins,facecolor='blue',alpha=0.5,label="Predicted vs Labels")
    plt.xlabel('Multinomial Negative LL Profile Labels and Preds in Probability Space')
    plt.title("MNNLL:"+title)
    plt.legend(loc='best')
    plt.savefig(outf_prefix+".mnnll.png",format='png',dpi=300)


    #plot jsd histogram    
    plt.figure()
    n,bins,patches=plt.hist(region_jsd,num_bins,facecolor='blue',alpha=0.5,label="Predicted vs Labels")
    if prep_jsd is not None:
        n2,bins2,patches2=plt.hist(pseudorep_jsd,num_bins,facecolor='red',alpha=0.5,label="Pseudoreps")
    n3,bins3,patches3=plt.hist(shuffled_labels_jsd,num_bins,facecolor='black',alpha=0.5,label='Labels vs Shuffled Labels')
    plt.xlabel('Jensen Shannon Distance Profile Labels and Preds in Probability Space')
    plt.title("JSD Dist.:"+title)
    plt.legend(loc='best')
    plt.savefig(outf_prefix+".jsd.png",format='png',dpi=300)
    if prep_jsd is not None:
        density_scatter(np.asarray(region_jsd),
                        np.asarray(pseudorep_jsd),
                        xlab='JSD Predict vs Labels',
                        ylab='JSD Pseudoreps',
                        title='JSD vs Pseudoreps:'+title,
                        figtitle=outf_prefix+".jsd.pseudorep.png")

    
    #get mean and std
    if len(pseudorep_jsd)>0:
        return nanmean(region_jsd), nanstd(region_jsd), nanmean(mnnll_vals), nanstd(mnnll_vals), nanmean(pseudorep_jsd), nanstd(pseudorep_jsd), nanmean(shuffled_labels_jsd), nanstd(shuffled_labels_jsd)
    else:
        return nanmean(region_jsd), nanstd(region_jsd), nanmean(mnnll_vals), nanstd(mnnll_vals), None, None, nanmean(shuffled_labels_jsd), nanstd(shuffled_labels_jsd)
    
def get_performance_metrics_profile_wrapper(args):
    if type(args)==type({}):
        args=config.args_object_from_args_dict(args)
    labels_and_preds={}
    bad_regions=[]
    for loss_index in range(len(args.losses)):
        cur_loss=args.losses[loss_index]
        cur_loss_suffix=args.loss_suffixes[loss_index]
        cur_pred=pd.read_hdf(args.predictions+"."+cur_loss_suffix,header=None,sep='\t')
        cur_labels=pd.read_hdf(args.labels+"."+cur_loss_suffix,header=None,sp='\t')
        if args.pseudoreps is not None:
            pseudoreps=[pyBigWig.open(rep) for rep in args.pseudoreps]
        else:
            pseudoreps=None
        labels_and_preds[cur_loss]={}
        labels_and_preds[cur_loss]['labels']=cur_labels
        labels_and_preds[cur_loss]['predictions']=cur_pred
        if cur_loss=="counts":
            #filter for regions of low or excessively high counts
            if (args.label_min_to_score is not None) or (args.label_max_to_score is not None):
                for index,row in cur_labels.iterrows():
                    cur_count=row[0]
                    if args.label_min_to_score is not None:
                        if cur_count < args.label_min_to_score:
                            bad_regions.append(index)
                    if args.label_max_to_score is not None:
                        if cur_count > args.label_max_to_score:
                            bad_regions.append(index)
    print("loaded labels and predictions")
    #drop bad regions
    for loss in labels_and_preds:
        labels_and_preds[loss]['labels']=labels_and_preds[loss]['labels'].drop(bad_regions)
        labels_and_preds[loss]['predictions']=labels_and_preds[loss]['predictions'].drop(bad_regions)
    print("removed regions with too few or too many counts")
    spearman_cor,pearson_cor, mse, spearman_cor_ps,pearson_cor_ps=counts_metrics(labels_and_preds['counts']['labels'],
                                                                                 labels_and_preds['counts']['predictions'],
                                                                                 args.outf,args.title,pseudoreps,args.flank)
    
    mean_jsd, std_jsd, mean_nll, std_nll, mean_pr_jsd, std_pr_jsd , mean_shuf_jsd, std_shuf_jsd = profile_metrics(labels_and_preds['profile']['labels'],
                                                                                    labels_and_preds['profile']['predictions'],
                                                                                    labels_and_preds['counts']['labels'],
                                                                                    labels_and_preds['counts']['predictions'],
                                                                                    args.outf,args.title,pseudoreps,args.flank,
                                                                                    args.smooth_observed_profile,
                                                                                    args.smooth_predicted_profile,
                                                                                    args.smooth_preps)
    outf=open(args.outf+".summary.txt",'w')
    outf.write('Title\tPearson\tSpearman\tMSE\tPseudorepPearson\tPseudorepSpearman\tMeanJSD\tStdJSD\tMeanPseudorepJSD\tStdPseudorepJSD\tMeanMNNLL\tStdMNNLL\tMeanShuffledJSD\tStdShuffledJSDn')
    outf.write(args.title+'\t'+str(pearson_cor)+'\t'+str(spearman_cor)+'\t'+str(mse)+'\t'+str(pearson_cor_ps)+'\t'+str(spearman_cor_ps)+'\t'+str(mean_jsd)+'\t'+str(std_jsd)+'\t'+str(mean_pr_jsd)+'\t'+str(std_pr_jsd)+'\t'+str(mean_nll)+'\t'+str(std_nll)+'\t'+str(mean_shuf_jsd)+'\t'+str(std_shuf_jsd)+'\n')
    outf.close() 

def main():
    args=parse_args()
    get_performance_metrics_profile_wrapper(args)
    
if __name__=="__main__":
    main()
    



    
