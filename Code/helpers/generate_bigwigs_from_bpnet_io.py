import pyBigWig
import operator 
from .transform_bpnet_io import *
import argparse
import h5py
import pdb

def parse_args():
    parser=argparse.ArgumentParser(description="generate bigwig files for visualizing bpnet labels & predictions")
    parser.add_argument("--predictions_hdf")
    parser.add_argument("--out_prefix")
    parser.add_argument("--bigwig_types",nargs="+",default=["labels_counts",
                                                            "labels_logits",
                                                            "labels_prob",
                                                            "predictions_counts",
                                                            "predictions_logits",
                                                            "predictions_prob",
                                                            "delta_logits",
                                                            "delta_prob"])
    parser.add_argument("--chrom_sizes_file")
    return parser.parse_args()

def make_bigwig(bw_type,bigwig_inputs,coords,coord_dict,args):
    chromsizes=[i.split('\t') for i in open(args.chrom_sizes_file,'r').read().strip().split('\n')]
    chromsizes=[tuple([i[0],int(i[1])]) for i in chromsizes]
    chromsizes=sorted(chromsizes,key=operator.itemgetter(0))
    print(chromsizes) 
    bw=pyBigWig.open('.'.join([args.out_prefix,bw_type,'bw']),'w')
    bw.addHeader(chromsizes)
    #add entries to the file
    last_end=0
    last_chrom=None
    for coord in coords:
        i=coord_dict[coord]
        cur_vals=bigwig_inputs[i,:]
        num_vals=cur_vals.shape[0]
        flank=num_vals//2
        cur_chrom=coord[0]
        cur_summit=int(coord[1])
        cur_start=cur_summit-flank
        if (last_chrom==cur_chrom) and (cur_start < last_end):
            delta=last_end-cur_start
            cur_vals=cur_vals[delta::]
            cur_start=last_end 
        cur_end=cur_summit+flank
        last_end=cur_end
        last_chrom=cur_chrom
        if len(cur_vals)==0:
            continue
        
        #print(cur_chrom+":"+str(cur_start)+"-"+str(cur_end)+";"+str(len(cur_vals)))
        bw.addEntries(cur_chrom, cur_start,values=cur_vals,span=1,step=1)
    bw.close()        
    return

def main():
    args=parse_args()
    preds=h5py.File(args.predictions_hdf,'r')
    coords=preds['coords'][:]
    coords=[tuple([i.decode('utf8')  for i in j[0:2]]) for j in coords]
    coords=[tuple([i[0],int(i[1])]) for i in coords] 
    print("loaded coords")
    bigwig_inputs=get_model_outputs_to_plot(preds)
    print("got bigwig inputs")
    #we must sort the coordinates so bigwig entries are written in order
    coord_dict={}
    for i in range(len(coords)):
        cur_coord=coords[i]
        coord_dict[cur_coord]=i
    coords=sorted(coords,key=operator.itemgetter(0,1))
    print(coords[0:20])
    for bw_type in args.bigwig_types:
        #generate bigwig
        make_bigwig(bw_type,bigwig_inputs[bw_type],coords,coord_dict,args)
        print("generated bigwig:"+str(bw_type) )
        
    
if __name__=="__main__":
    main()
    
