#!/bin/bash

dir=$(realpath $(dirname "$0"))
preset=$1
name=${2:-$preset}

$dir/clean_runs.sh $name
python wishlist.py --preset $preset -o --folder $name && python main.py --folder $name