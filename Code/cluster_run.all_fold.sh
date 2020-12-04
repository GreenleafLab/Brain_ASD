#!/bin/bash
fold="{fold number}"
outdir=.
model="{cluster name}"
seed="{seed val}"
gpu=0
title='bias corrected '
./cluster_train.sh $fold $gpu $model $seed $outdir
./cluster_predict.sh $fold $gpu $model $seed $outdir
./cluster_score.sh $outdir $model $fold $title 
