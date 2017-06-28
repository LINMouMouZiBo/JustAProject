#!/bin/bash

for file in "$1"/*.avi
do
	avconv -i "$file" -c:v libx264 -c:a copy "${file[@]%.avi}".mp4
done
