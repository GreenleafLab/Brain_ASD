#!/bin/bash
fold=1
outdir=.
model=c13
seed=1234
gpu=5
title='bias corrected '
train/c13_train.sh $fold $gpu $model $seed $outdir
predict/c13_predict.sh $fold $gpu $model $seed $outdir
score/c13_score.sh $outdir $model $fold $title 
