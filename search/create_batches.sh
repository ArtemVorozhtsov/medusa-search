#!/bin/bash
: '
    arguments:
    1 - number of shards
    2 - upper limit filesize
    3 - inner limit filesize   
'
# mkdir batches

hrms_database_path=$(niet .hrms_database_path ../config.yaml)

all_spec_n=$(find $hrms_database_path -name "*.mzXML" -size -"$2" -size +"$3" | wc -l)
all_spec_n=$(expr $all_spec_n + 30)
batch_size=$(expr $all_spec_n / $1)
find ~/maxis_data/converted/ -name "*.mzXML" -size +"$3" -size -"$2" | (cd batches; split -l $batch_size)

