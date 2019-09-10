#!/usr/bin/env bash

for ((x=0; x<5; x++));
do
    echo "Running $x";
    python analyse-features.py $x;
done

