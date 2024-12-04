#!/usr/bin/bash
# 1 - Csv-filename
# 2 - n_jobs

while IFS="," read -r compound_name formula
do
  echo "Compound name: $compound_name"
  echo "Formula: $formula"
  
  mkdir ../search/reports/$compound_name
   
  for batch in ../search/index_pickles/*.pkl
      do
          python ../search/search.py $formula 1 $batch $2 $compound_name No auto Yes 60 No
      done
  
  if [[ $(cat ../search/reports/$compound_name/*.csv) ]]; then
    cat ../search/reports/$compound_name/*.csv | sort -k3 -t',' -g > ../search/reports/$compound_name/combined.csv
  else
    echo "no ions found"
    rm -rf ../search/reports/$compound_name
  fi
    echo ""

done < $1

echo "Search finished. You may check results." | python notify_finish.py
