#!/bin/bash
fold=0
outdir=.
model=c15
seed=1234
gpu=6
title='bias corrected '
train/c15_train.sh $fold $gpu $model $seed $outdir
predict/c15_predict.sh $fold $gpu $model $seed $outdir 
score/c15_score.sh $outdir $model $fold $title 
