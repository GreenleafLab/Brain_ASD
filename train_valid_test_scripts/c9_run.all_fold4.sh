#!/bin/bash
fold=4
outdir=.
model=c9
seed=1234
gpu=4
title='bias corrected '
train/c9_train.sh $fold $gpu $model $seed $outdir
predict/c9_predict.sh $fold $gpu $model $seed $outdir 
score/c9_score.sh $outdir $model $fold $title 
