#!/bin/bash

perform_run() {
  local arg1=$1
  shift 1
  echo $arg1
}


presets=("cassini_bunch" "a" "b" "c" "d" "e" "f" "g" "h" "i" "j")

for preset in ${presets[@]}; do
  perform_run $preset
done