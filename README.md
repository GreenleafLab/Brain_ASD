# Brain_ASD

This repo contains scripts for training the neural network models used for prioritizing ASD mutations with the fetal brain ATAC-seq atlas.
Another repo for the raw dataset and single cell analysis used in the manuscript can be found here : https://github.com/GreenleafLab/brainchromatin

Used to train BPNET models in Trevino et al. 2020

Preprint here: https://www.biorxiv.org/content/10.1101/2020.12.29.424636v1

This code base uses two other repos : KerasAC (https://zenodo.org/record/4248179#.X8skj5NKiF0)  and seqdataloader (https://zenodo.org/record/3771365#.X8skqZNKiF0) as part of the data processing and model training scripts. 

Folder Structure :

cluster_peaks :
peaks_c*_bpnet.csv - Cluster specific peaks extended +/-500 bp around the summit.
peaks_all_bpnet.csv - Peaks called from pseudobulk of all cells across all timepoints.
The bigwigs used for model training can be obtained from the links.txt file in the repo : https://github.com/GreenleafLab/brainchromatin/blob/main/links.txt

gc_matched_peaks:
c*_matched_gc_fold*.csv - Cluster specific gc matched negative regions with 5 different matched negatives as 5 folds.

OtherFiles: 
bias.atac.24mer.*.weights.data-00000-of-00001 - Tobias weight tracks 
hg38.chrom.sizes - hg 38 chromosome sizes file 
GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta - hg38 fasta file used for training the model

train_valid_test_scripts:

params : contains the cluster specific weights for the mean squared error (MSE) loss on the log of the total counts and a multinomial negative log-likelihood loss (MNLL) for the profile probability output and fold specific tobias bias weights for the models.

run : contains the calls for train, predict and score files described below. It takes in parameters including the GPU , seed value for all random initializations, the working folder where the outputs weights and other files are to be stored and name for the models that will be used while storing the files. This file is self sufficient for training all the models . 

train : This folder contains shell scripts for training base pair resolution models for all clusters and takes in parameters that are set by the cluster and fold specific run.all.sh file .

predict :  This folder contains shell scripts for predicting base pair resolution models for all clusters and takes in parameters that are set by the cluster and fold specific run.all.sh file .

score :  This folder contains shell scripts for scoring base pair resolution models for all clusters for all the prediction metrics on the witheld test chromosomes and takes in parameters that are set by the cluster and fold specific run.all.sh file .



Working to make this public repository more clear and accessible. We will continue to make updates, so thank you for your patience. 

