#!/bin/bash
fold=1
outdir=.
model=c10
seed=1234
gpu=6
title='bias corrected '
train/c10_train.sh $fold $gpu $model $seed $outdir
predict/c10_predict.sh $fold $gpu $model $seed $outdir 
score/c10_score.sh $outdir $model $fold $title 
