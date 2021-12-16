#!/bin/bash

rm *.ad *.txt *~

FILE=./main
rm -i $FILE
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    make
fi

sbatch roda.sh
