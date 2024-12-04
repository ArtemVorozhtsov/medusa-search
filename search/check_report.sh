#!/bin/bash

cat reports/$1/combined.csv | sed 's/,/ ,/g' | column -t -s, 
