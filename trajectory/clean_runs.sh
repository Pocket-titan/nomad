#!/bin/bash
find ./runs -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

if [ -f ./runs/main.log ]; then
  rm ./runs/main.log
fi

if [ -f ./runs/wishlist.pkl ]; then
  rm ./runs/wishlist.pkl
fi

if [ -f ./runs/wishlist_after.pkl ]; then
  rm ./runs/wishlist_after.pkl
fi
