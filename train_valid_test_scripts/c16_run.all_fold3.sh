#!/bin/bash
fold=3
outdir=.
model=c16
seed=1234
gpu=6
title='bias corrected '
train/c16_train.sh $fold $gpu $model $seed $outdir
predict/c16_predict.sh $fold $gpu $model $seed $outdir
score/c16_score.sh $outdir $model $fold $title 
