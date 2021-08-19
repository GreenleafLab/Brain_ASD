#aggregate summary information for enzymatic bias models
import pandas as pd
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="aggregate summary statistics")
    parser.add_argument("--summary_stats_file")
    parser.add_argument("--performance_fields",nargs="+",default=["Pearson",
                                                                  "Spearman",
                                                                  "MSE",
                                                                  "PseudorepPearson",
                                                                  "PseudorepSpearman",
                                                                  "MeanJSD",
                                                                  "StdJSD",
                                                                  "MeanPseudorepJSD",
                                                                  "StdPseudorepJSD",
                                                                  "MeanMNNLL",
                                                                  "StdMNNLL"])
    parser.add_argument("--out")
    return parser.parse_args()

def main():
    args=parse_args()
    data=pd.read_csv(args.summary_stats_file,header=0,sep='\t')
    outf=open(args.out,'w')
    metadata_cols=list(data.columns)
    metadata_cols.remove('File') 
    metric_cols=args.performance_fields 
    header='\t'.join([str(i) for i in metadata_cols])+'\t'+'\t'.join([str(i) for i in metric_cols])
    outf.write(header+'\n')
    for index,row in data.iterrows():
        print(str(row['File']))
        perf=pd.read_csv(row['File'],header=0,sep='\t')
        data_vals=[str(row[i]) for i in metadata_cols]
        perf_vals=[str(perf[i][0]) for i in metric_cols]
        towrite=data_vals+perf_vals
        outf.write('\t'.join(towrite)+'\n')
    outf.close()

                        
if __name__=="__main__":
    main()
                        
