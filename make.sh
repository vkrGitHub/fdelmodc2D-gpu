#!/bin/bash

# Check input parameters
if [ "$#" -ne 1 ]; then
		echo "make.sh: wrong number of input parameters. Exiting."
		echo "make.sh install"
		echo "     Installs gnu parallel locally here, make fdelmodc2D-gpu, make install."
		echo "make.sh uninstall"
		echo "     Uninstalls gnu parallel, make clean fdelmodc2D-gpu"
		exit
fi

if [[ $1 == "install" ]]; then
	# Module loads
	module purge
	module load gcc/7.3.0 cuda/11.0

	# Install gnu parallel
	cd parallel_standalone
	root=$(pwd)
	#tar -xf parallel-latest.tar.bz2
	tar -xf PARALLEL-LATEST.TAR.BZ2
	cd parallel-20200722
	./configure --prefix=${root}/parallel_installed 
	make
	make install
	cd ..
	rm -r parallel-20200722
	cp ${root}/parallel_installed/bin/parallel ../../bin

	# Install fdelmodc2D-gpu
	cd $root; cd ..
	make
	make install
elif [[ $1 == "uninstall" ]]; then
	make clean
	rm -r parallel_standalone/parallel_installed
	rm ../bin/parallel ../bin/fdelmodc2D*
else
	echo "Invalid option. Please choose 'install' or 'uninstall' "
fi