#!/bin/bash
: '
    arguments:
    1 - word-indicator
    2 - batch filename
    3 - upper limit filesize
    4 - inner limit filesize   
'
hrms_database_path=$(niet .hrms_database_path ../config.yaml)

find $hrms_database_path -type d -iname "*$1*" -print0 | while read -d $'\0' folder
do 
    find "$folder" -name "*.mzXML" -size +"$4" -size -"$3" >> "batches/$2"  
done



