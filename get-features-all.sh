#!/usr/bin/env bash

FILES=$(ls data)
ext=".txt"
re="^[0-9]+$"
#re=".*"

for f in $FILES;
do
    id=${f%"$ext"}
    if [[ $id =~ $re ]] ; then
        echo "Running $id";
        python get-features.py $id;
    fi  
done

