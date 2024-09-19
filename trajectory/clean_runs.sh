#!/bin/bash
folder=${1:-runs}
folder=$(realpath $folder)


find $folder -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

if [ -f $folder/main.log ]; then
  rm $folder/main.log
fi

if [ -f $folder/wishlist.pkl ]; then
  rm $folder/wishlist.pkl
fi

if [ -f $folder/wishlist_after.pkl ]; then
  rm $folder/wishlist_after.pkl
fi
