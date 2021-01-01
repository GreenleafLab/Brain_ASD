#!/bin/bash
fold=1
outdir=.
model=c7
seed=1234
gpu=7
title='bias corrected '
train/c7_train.sh $fold $gpu $model $seed $outdir
predict/c7_predict.sh $fold $gpu $model $seed $outdir 
score/c7_score.sh $outdir $model $fold $title 
