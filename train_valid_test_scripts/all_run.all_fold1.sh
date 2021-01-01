#!/bin/bash
fold=1
outdir=.
model=all
seed=1234
gpu=1
title='bias corrected '
train/all_train.sh $fold $gpu $model $seed $outdir
predict/all_predict.sh $fold $gpu $model $seed $outdir 
score/all_score.sh $outdir $model $fold $title 
