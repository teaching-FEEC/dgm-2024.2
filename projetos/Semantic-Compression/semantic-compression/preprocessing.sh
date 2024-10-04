#!/bin/bash

mkdir data
mkdir data/train
mkdir data/val

python3 dividir.py

cd data


find . -name "*.jpg" |  parallel -j64 convert -verbose {} -resize "256x256^" {}

find . -name "*.jpg" |  parallel -j64 convert -verbose {} -resize "256x256^" {}


find . -name "*.jpg" |  parallel -j64 convert -verbose {} -crop "256x256^" {}

find . -name "*.jpg" |  parallel -j64 convert -verbose {} -crop "256x256^" {}

cd train/

find  -type f -name "*.png" -o -name "*.jpg" | while read img; do   size=$(identify -format "%wx%h" "$img");   if [ "$size" != "256x256" ]; then     rm "$img";   fi; done

cd ..


cd val/

find  -type f -name "*.png" -o -name "*.jpg" | while read img; do   size=$(identify -format "%wx%h" "$img");   if [ "$size" != "256x256" ]; then     rm "$img";   fi; done
