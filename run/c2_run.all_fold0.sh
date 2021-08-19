#!/bin/bash
fold=0
outdir=.
model=c2
seed=1234
gpu=2
title='bias corrected '
train/c2_train.sh $fold $gpu $model $seed $outdir
predict/c2_predict.sh $fold $gpu $model $seed $outdir
score/c2_score.sh $outdir $model $fold $title 
