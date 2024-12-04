#!/bin/bash
: '
    arguments: 1 - window_size, 2 - min_distance
'
for batch  in ./batches/*
    do
        filename=$(basename -- "$batch")
        filename="${filename%.*}"
        echo $filename
        FILE="./index_pickles/${filename}.pkl"
        if [ -f "$FILE" ]; then
            echo "$FILE exists."
        else
            cat  $batch  | python process.py $batch $1 $2 
        fi
    done
