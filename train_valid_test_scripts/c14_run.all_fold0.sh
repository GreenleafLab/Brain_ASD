#!/bin/bash
fold=0
outdir=.
model=c14
seed=1234
gpu=1
title='bias corrected '
train/c14_train.sh $fold $gpu $model $seed $outdir
predict/c14_predict.sh $fold $gpu $model $seed $outdir 
score/c14_score.sh $outdir $model $fold $title 
