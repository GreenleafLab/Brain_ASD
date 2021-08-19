#!/bin/bash
fold=$1

gpu=$2

#create a title for the model
model_name=$3

#set seed for training
if [ -z "$4" ]
then
    seed=1234
else
    seed=$4
fi
echo "seed:$seed"

#output directory 
if [ -z "$5" ]
then
    outdir='.'
else
    outdir=$5
fi
echo "outdir:$outdir"
CUDA_VISIBLE_DEVICES=$gpu kerasAC_predict_tdb \
		    --batch_size 20 \
		    --ref_fasta /home/lakss/misc_files/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --tdb_array /home/lakss/tiledb/db \
		    --tdb_partition_attribute_for_upsample peak \
		    --tdb_partition_thresh_for_upsample 2 \
		    --tdb_partition_datasets_for_upsample $model_name \
		    --tdb_input_source_attribute seq \
		    --tdb_input_aggregation None \
		    --tdb_input_transformation None \
		    --tdb_input_flank 1057 \
		    --tdb_output_source_attribute count_bigwig_unstranded_5p count_bigwig_unstranded_5p \
		    --tdb_output_flank 500 500 \
		    --tdb_output_aggregation None sum \
		    --tdb_output_transformation None log \
		    --tdb_input_datasets seq \
		    --tdb_output_datasets $model_name $model_name \
		    --num_inputs 1 \
		    --num_outputs 2 \
		    --chrom_sizes hg38.chrom.sizes \
		    --tiledb_stride 50 \
		    --fold $fold \
		    --genome hg38 \
		    --upsample_ratio_list_predict 1 \
		    --predictions_and_labels_hdf5 $outdir/$model_name.$fold \
		    --json $model_name.$fold.arch \
		    --weights $model_name.$fold.weights \
		    --upsample_threads 1 \
		    --tdb_ambig_attribute ambig_peak \
		    --tdb_transformation_pseudocount 1
