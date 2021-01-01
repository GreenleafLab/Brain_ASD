#!/bin/bash
fold=4
outdir=/srv/scratch/lakss/brain_organoid/bias_corrected_bpnet/
model=c8
seed=1234
gpu=7
title='bias corrected '
train/c8_train.sh $fold $gpu $model $seed $outdir
predict/c8_predict.sh $fold $gpu $model $seed $outdir 
score/c8_score.sh $outdir $model $fold $title 
