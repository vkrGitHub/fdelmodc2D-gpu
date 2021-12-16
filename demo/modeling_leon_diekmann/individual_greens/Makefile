FC    = gfortran -ffree-line-length-none
PROG  = main
MODS  = mdle_io.f90
FLAGS = -O3 -fopenmp -fcheck=all

all:
	$(FC) $(MODS) -c $(FLAGS)
	$(FC) -c $(PROG).f90 $(FLAGS)
	$(FC) *.o -o $(PROG) $(FLAGS)
	rm *.o *.mod

clean:
	rm ./$(PROG) *~
