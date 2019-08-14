#!/bin/bash
#

path='temp/mol_name'
total=$(echo ${path}/*.csv | wc -w)

for i in ${path}/*.csv ; do
    head -1 $i > temp/atom-new.csv
    break
done

for i in $(find ${path} -name '*.csv' | tqdm --total ${total} --unit=files); do
    tail -n +2 $i >> temp/atom-new.csv
done
