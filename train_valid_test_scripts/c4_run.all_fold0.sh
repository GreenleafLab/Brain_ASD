#!/bin/bash
fold=0
outdir=/srv/scratch/lakss/brain_organoid/bias_corrected_bpnet/
model=c4
seed=1234
gpu=3
title='bias corrected '
train/c4_train.sh $fold $gpu $model $seed $outdir
predict/c4_predict.sh $fold $gpu $model $seed $outdir 
score/c4_score.sh $outdir $model $fold $title 
