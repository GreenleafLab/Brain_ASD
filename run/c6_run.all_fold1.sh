#!/bin/bash
fold=1
outdir=.
model=c6
seed=1234
gpu=2
title='bias corrected '
train/c6_train.sh $fold $gpu $model $seed $outdir
predict/c6_predict.sh $fold $gpu $model $seed $outdir 
score/c6_score.sh $outdir $model $fold $title 
