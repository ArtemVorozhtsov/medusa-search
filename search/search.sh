#!/bin/bash

mkdir reports/$5

if [[ "${10}" == "Yes" ]]; then
    mkdir reports/$5/plots
fi

for batch in $3/*.pkl
    do
        python search.py $1 $2 $batch $4 $5 $6 $7 $8 $9 ${10}
    done

if [[ $(cat reports/$5/*.csv) ]]; then
    cat reports/$5/*.csv | sort -k3 -t',' -g > reports/$5/combined.csv
else
    echo "no ions found"
    rm -rf reports/$5
fi

