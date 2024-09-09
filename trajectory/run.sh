#!/bin/sh

dir=$(realpath $(dirname "$0"))

$dir/clean_runs.sh
python $dir/wishlist.py unpowered EMJN ultra
python $dir/main.py