#!/bin/bash

# Check IO
if [ "$#" -ne 1 ]; then
	echo "singrun.sh: wrong number of input parameters. Exiting."
	echo -e "Usage:\n singrun.sh sbatch_script.sh"
	exit
fi

# 1st: SBATCHs header
sed '/^#SBATCH.*/!d' $1 >./jobs/file1_$1

# 2nd: 
# singularity header: load bashrc, load modules
rm ./jobs/stripped_$1
echo '#!/bin/bash' >>./jobs/stripped_$1
echo "source $HOME/.bashrc" >>./jobs/stripped_$1
echo "module load SeisUnix/44R2 gcc/8.3.1" >>./jobs/stripped_$1
echo "source $HOME/.bashrc" >>./jobs/stripped_$1
# get file content without SBATCHs
sed '/^#SBATCH.*/d' $1 >>./jobs/stripped_$1
chmod 777 ./jobs/stripped_$1

# 3rd: singularity exec line
echo "singularity exec -B /scratch ${CONTAINER_MARCH} ./jobs/stripped_$1" >./jobs/file2_$1

# 4th: compose file that will be sbatched
rm ./jobs/file_sbatch_$1
echo '#!/bin/bash' >>./jobs/file_sbatch_$1
cat ./jobs/file1_$1 >>./jobs/file_sbatch_$1
cat ./jobs/file2_$1 >>./jobs/file_sbatch_$1
echo -e "file_sbatch is \n$(cat ./jobs/file_sbatch_$1)\n"
chmod 777 ./jobs/file_sbatch_$1

# 5th: sbatch the file
sbatch ./jobs/file_sbatch_$1

# Cleanup (do not erase stripped or file_sbatch)
rm ./jobs/file1_$1 ./jobs/file2_$1
