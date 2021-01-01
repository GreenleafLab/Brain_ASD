#!/bin/bash
fold=2
outdir=.
model=c12
seed=1234
gpu=1
title='bias corrected '
train/c12_train.sh $fold $gpu $model $seed $outdir
predict/c12_predict.sh $fold $gpu $model $seed $outdir
score/c12_score.sh $outdir $model $fold $title 
