#!/bin/bash
fold=1
outdir=.
model=c11
seed=1234
gpu=3
title='bias corrected '
train/c11_train.sh $fold $gpu $model $seed $outdir
predict/c11_predict.sh $fold $gpu $model $seed $outdir 
score/c11_score.sh $outdir $model $fold $title 
