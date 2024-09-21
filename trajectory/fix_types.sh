#!/bin/bash

pkg=$1
# pybind11-stubgen tudatpy
# pybind11-stubgen pygmo
# pybind11-stubgen pygmo_plugins_nonfree
# stubgen -p godot -o stubs --include-docstrings
# cp -RT stubs/tudatpy/ ~/miniconda3/envs/tudat-space/lib/python3.9/site-packages/tudatpy/
# cp -RT stubs/pygmo/ ~/miniconda3/envs/tudat-space/lib/python3.9/site-packages/pygmo/
# cp -RT stubs/pygmo_plugins_nonfree/ ~/miniconda3/envs/tudat-space/lib/python3.9/site-packages/pygmo_plugins_nonfree/
# cp -RT stubs/godot/ ~/miniconda3/envs/tudat-space/lib/python3.9/site-packages/godot/

stubgen -p $pkg -o stubs --include-docstrings
cp -RT stubs/$pkg/ ~/miniconda3/envs/tudat-space/lib/python3.9/site-packages/$pkg/