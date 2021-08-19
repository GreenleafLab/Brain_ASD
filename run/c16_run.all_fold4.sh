#!/bin/bash
fold=0
outdir=.
model=c16
seed=1234
gpu=7
title='bias corrected '
./c16_train.sh $fold $gpu $model $seed $outdir
./c16_predict.sh $fold $gpu $model $seed $outdir
./c16_score.sh $outdir $model $fold $title 
