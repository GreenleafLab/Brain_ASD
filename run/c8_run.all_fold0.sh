#!/bin/bash
fold=0
outdir=.
model=c8
seed=1234
gpu=3
title='bias corrected '
train/c8_train.sh $fold $gpu $model $seed $outdir
predict/c8_predict.sh $fold $gpu $model $seed $outdir 
score/c8_score.sh $outdir $model $fold $title 
