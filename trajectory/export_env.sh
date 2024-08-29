#!/bin/bash
conda env export --name tudat-space --no-builds >environment.yml
sed -i.bak '/^prefix:/d' environment.yml

PACKAGES=("zlib" "libzlib" "scotch")
PATTERN=$(
  IFS="|"
  echo "${PACKAGES[*]}"
)

sed -i.bak -E "/^\s*-\s*(${PATTERN})=/s/=.*$//" environment.yml
sed -i.bak -E "/^\s*-\s*(${PATTERN})==/s/==.*$//" environment.yml
rm environment.yml.bak
