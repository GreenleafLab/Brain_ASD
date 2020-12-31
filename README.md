# Brain_ASD

This repo contains scripts for training the neural network models used for prioritizing ASD mutations with the fetal brain ATAC-seq atlas

Folder Structure :

cluster_peaks :
peaks_c*_bpnet.csv - Cluster specific peaks extended +/-500 bp around the summit.
peaks_all_bpnet.csv - Peaks called from pseudobulk of all cells across all timepoints.

gc_matched_peaks:
c*_matched_gc_fold*.csv - Cluster specific gc matched negative regions with 5 different matched negatives as 5 folds.

OtherFiles: 
bias.atac.24mer.*.weights.data-00000-of-00001 - Tobias weight tracks 
hg38.chrom.sizes - hg 38 chromosome sizes file 
GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta - Hg38 fasta file used for training the model

All weight files of trained neural network models are available in the google drive below.
https://drive.google.com/drive/folders/1SrpUKl51AsBG7e5S1ZCkEAxMTSfr76yK?usp=sharing
