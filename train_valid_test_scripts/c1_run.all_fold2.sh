#!/bin/bash
fold=2
outdir=.
model=c1
seed=1234
gpu=7
title='bias corrected '
train/c1_train.sh $fold $gpu $model $seed $outdir
score/c1_predict.sh $fold $gpu $model $seed $outdir
predict/c1_score.sh $outdir $model $fold $title 
