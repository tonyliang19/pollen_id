#!/bin/bash


# for loop to run the standlone script as many times of arguments provided
for (( b=1; b<=5; b++))
do
	echo "Submmtting for bundle $b"
	sh pollen_infer.sh "bun$b" "predict"
	echo "Done for bundle $b"
done
