#!/bin/bash
fold=2
outdir=/srv/scratch/lakss/brain_organoid/bias_corrected_bpnet/
model=c5
seed=1234
gpu=6
title='bias corrected '
train/c5_train.sh $fold $gpu $model $seed $outdir
predict/c5_predict.sh $fold $gpu $model $seed $outdir 
score/c5_score.sh $outdir $model $fold $title 
