#!/bin/bash

for file in `ls | grep 000`
do
    less "${file}/logfile.txt"
    echo ${file}
    #less "${file::-1}/logfile.txt"
    #echo ${file::-1}
    read -p "Remove this folder? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
	#mv ${file::-1} trash/
	mv ${file} trash/
    fi
done
