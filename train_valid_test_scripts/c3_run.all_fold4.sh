#!/bin/bash
fold=4
outdir=.
model=c3
seed=1234
gpu=2
title='bias corrected '
train/c3_train.sh $fold $gpu $model $seed $outdir
predict/c3_predict.sh $fold $gpu $model $seed $outdir
score/c3_score.sh $outdir $model $fold $title 
