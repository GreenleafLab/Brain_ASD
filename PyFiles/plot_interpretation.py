import argparse
import numpy as np 
import pdb
from concise.utils.plot import seqlogo, seqlogo_fig
import pysam
import argparse
import pyBigWig
import matplotlib.pyplot as plt

def parse_args():
    parser=argparse.ArgumentParser(description="generate plot of sequence importance scores")
    parser.add_argument("--input_bigwig")
    parser.add_argument("--chrom",nargs="+")
    parser.add_argument("--startpos",nargs="+",type=int)
    parser.add_argument("--endpos",nargs="+",type=int)
    parser.add_argument("--outf",default=None)
    parser.add_argument("--outf_width",type=float,default=3)
    parser.add_argument("--outf_length",type=float,default=25)
    parser.add_argument("--ref")
    parser.add_argument("--minthresh",type=float,default=None)
    parser.add_argument("--maxthresh",type=float,default=None) 
    return parser.parse_args() 
def one_hot_encode_sequence(chrom,start,end,ref):    
    num_generated=0
    ref=pysam.FastaFile(ref)
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    seq=ref.fetch(chrom,start,end)
    seq=np.array([ltrdict.get(x,[0,0,0,0]) for x in seq])
    return seq

def plot_seq_importance(scores, data, ylim=None, figsize=(25, 3),outf=None,tick_interval=5):
    seq_len = data.shape[0]
    product=np.expand_dims(scores,axis=1)*data
    product=np.nan_to_num(product,copy=False)
    print(product.shape)
    seqlogo_fig(product, figsize=figsize)
    plt.xticks(list(range(0, product.shape[0], tick_interval)))
    if ylim!=None:
        plt.ylim(ylim)
    if outf==None:
        plt.show()
    else:
        plt.axis("off")
        plt.gca().set_position([0,0,1,1])
        plt.savefig(outf)

def main():
    args=parse_args()
    for i in range(len(args.startpos)):
        cur_chrom=args.chrom[i]
        cur_startpos=args.startpos[i]
        cur_endpos=args.endpos[i]
        cur_data=one_hot_encode_sequence(cur_chrom,cur_startpos,cur_endpos,args.ref)
        #extract the signal from the bigwig input file
        bw=pyBigWig.open(args.input_bigwig)
        scores=bw.values(cur_chrom,cur_startpos,cur_endpos)
        if args.minthresh!=None:
            scores=[max([i,args.minthresh]) for i in scores]
        if args.maxthresh!=None:
            scores=[max([i,args.maxthresh]) for i in scores]
            
        #get the one-hot-encoded sequence in this interval
        seq=one_hot_encode_sequence(cur_chrom,cur_startpos,cur_endpos,args.ref)
        #generate the plot for the current interval
        plot_seq_importance(scores, seq, figsize=(args.outf_length, args.outf_width),outf=args.outf)

if __name__=="__main__":
    main()
    
