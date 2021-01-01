#!/bin/bash
fold=0
outdir=.
model=all
seed=1234
gpu=0
title='bias corrected '
train/all_train.sh $fold $gpu $model $seed $outdir
predict/all_predict.sh $fold $gpu $model $seed $outdir 
score/all_score.sh $outdir $model $fold $title 
