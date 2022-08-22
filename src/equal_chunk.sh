#!/bin/bash

directory=${1:-to_annotate}
sizelimit=${2:-1000}
sizesofar=0
dircount=1

du -s --block-size=1M "$directory"/* | while read -r size file
do 
    if ((sizesofar + size > sizelimit))
    then
        (( dircount ++ ))
        sizesofar=0
    fi
    (( sizesofar += size ))
    mkdir -p -- "$directory/sub_$dircount"
     mv -- "$file" "$directory/sub_$dircount"                                           
done 
