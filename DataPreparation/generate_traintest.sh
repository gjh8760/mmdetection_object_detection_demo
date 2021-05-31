#!/bin/sh

cd ../data/VOC/Annotations/
mkdir ../ImageSets
mkdir ../ImageSets/Main
ls -1 | sed -e 's/\.xml$//' | sort -n > ../ImageSets/Main/whole.txt

