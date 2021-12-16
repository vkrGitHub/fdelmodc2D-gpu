#!/bin/bash

# RUN WITH:
#time parallel ./gnu_parallel_test.sh {1} 1\>slurm_{= {1} =}.out 2\>slurm_{= {1} =}.err ::: 0 1 2 3
#INFO:
#https://www.gnu.org/software/parallel/parallel_tutorial.html

# Check IO
if [ "$#" -ne 1 ]; then
	echo ".sh: wrong number of input parameters. Exiting."
	echo -e "Usage:\n test.sh jobnum"
	exit
fi

echo "arg1=$1"
echo "I will run on GPU $1"
sleep 5
echo "Job $1 done sleeping 5"

# Testing stderr
echo "TESTING STDERR HERE AT JOB $1!!" 1>&2
