#!/bin/bash
fold=4
outdir=.
model=c13
seed=1234
gpu=0
title='bias corrected '
train/c13_train.sh $fold $gpu $model $seed $outdir
predict/c13_predict.sh $fold $gpu $model $seed $outdir
score/c13_score.sh $outdir $model $fold $title 
