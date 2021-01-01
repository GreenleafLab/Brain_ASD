#!/bin/bash
fold=1
outdir=.
model=c0
seed=1234
gpu=1
title='bias corrected '
train/c0_train.sh $fold $gpu $model $seed $outdir
predict/c0_predict.sh $fold $gpu $model $seed $outdir
score/c0_score.sh $outdir $model $fold $title 
