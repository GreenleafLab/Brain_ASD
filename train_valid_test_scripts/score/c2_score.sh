outdir=$1
model_name=$2
fold=$3
title=$4
kerasAC_score_bpnet \
    --predictions $outdir/$model_name.$fold.predictions \
    --losses profile counts \
    --outf $outdir/$model_name.$fold.scores \
    --title $title \
    --label_min_to_score 2.3 \
    --label_max_to_score 11.5 \
    --num_tasks 1
