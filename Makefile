# Makefile

include ../Make_include

########################################################################
# define general include and system library
ALLINC  = -I.
#LIBS    += -L$L -lgenfft -lm $(LIBSM)
#OPTC = -g -Wall -fsignaling-nans -O0 -fopenmp
#OPTC += -fopenmp -Waddress
#OPTC := $(subst -O3 -ffast-math, -O1 -g ,$(OPTC))
#PGI options for compiler feedback
#OPTC += -Mprof=lines
#OPTC += -qopt-report
#LDFLAGS += -Mprof=lines
CUFLAGS= -ftz=false -prec-div=true -prec-sqrt=true

#all: fdelmodc fdelmodc-multigpu fdelmodc-multicore
all: fdelmodc2D_singleGPU_test fdelmodc2D_singleGPU fdelmodc2D_multicore fdelmodc2D_multiGPU

PRG = fdelmodc2D_singleGPU_test

MAIN = $(PRG).cu
MAIN2 = fdelmodc2D_singleGPU.cu
MAIN3 = fdelmodc2D_multiGPU.cu
MAIN4 = fdelmodc2D_multicore.cu

SRCC	= acoustic2.c \
		acoustic4.c \
		acousticSH4.c \
		acoustic4_qr.c \
		acoustic6.c \
		viscoacoustic4.c \
		elastic4.c \
		elastic4dc.c \
		elastic6.c \
		viscoelastic4.c \
		defineSource.c  \
		getParameters.c  \
		getWaveletInfo.c  \
		getModelInfo.c  \
		applySource.c  \
		getRecTimes.c  \
		getBeamTimes.c  \
		writeSnapTimes.c  \
		writeRec.c  \
		writeSrcRecPos.c  \
		decomposition.c  \
		fileOpen.c  \
		recvPar.c  \
		readModel.c  \
		sourceOnSurface.c  \
		getWaveletHeaders.c  \
		boundaries.c  \
		verbosepkg.c  \
		writesufile.c  \
		gaussGen.c  \
		spline3.c  \
		CMWC4096.c  \
		wallclock_time.c  \
		atopkge.c \
		docpkge.c \
		threadAffinity.c \
		getpars.c  

MOBJ	= $(MAIN:%.cu=%.o)
MOBJ2	= $(MAIN2:%.cu=%.o)
MOBJ3	= $(MAIN3:%.cu=%.o)
MOBJ4	= $(MAIN4:%.cu=%.o)
OBJC	= $(SRCC:%.c=%.o)
OBJCU	= cuda_myutils.o \
			cuda_fileOpen.o \
			cuda_fdel2d_parameters.o \
			cuda_acoustic4.o \
			cuda_applySource.o \
			cuda_boundaries.o \
			cuda_sourceOnSurface.o \
			cuda_getRecTimes.o \
			cuda_writeRec.o \
			cuda_writeSnapTimes.o

# Prints some info to user
spacer=-------------------------------------------------------
$(warning MOBJ is $(MOBJ))
$(warning $(spacer))
$(warning MOBJ2 is $(MOBJ2))
$(warning $(spacer))
$(warning OBJC is $(OBJC))
$(warning $(spacer))
$(warning OBJCU is $(OBJCU))
$(warning $(spacer))
$(warning LIBS is $(LIBS))
$(warning $(spacer))
$(warning LDFLAGS is $(LDFLAGS))
$(warning $(spacer))
$(warning CFLAGS is $(CFLAGS))
$(warning $(spacer))
$(warning CUFLAGS is $(CUFLAGS))

# Multi-GPU executable
# fdelmodc-multigpu: $(MOBJ2) $(OBJC) fdelmodc.h $(OBJCU)
# 	nvcc $(LDFLAGS) $(CFLAGS) $(CUFLAGS) -Xcompiler "$(OPTC)" -o $@ $(MOBJ) $(OBJC) $(LIBS) $(OBJCU)
# todo: multicore manyshots
# fdelmodc2D_multicore: $(MOBJ4) $(OBJC) $(OBJCU)
# 	nvcc $(LDFLAGS) $(CFLAGS) $(CUFLAGS) -Xcompiler "$(OPTC)" -o $@ $(MOBJ3) $(OBJC) $(LIBS) $(OBJCU)

# todo: multiGPU
# fdelmodc2D_multiGPU: $(MOBJ3) $(OBJC) $(OBJCU)
# 	nvcc $(LDFLAGS) $(CFLAGS) $(CUFLAGS) -Xcompiler "$(OPTC)" -o $@ $(MOBJ2) $(OBJC) $(LIBS) $(OBJCU)

# todo: singleGPU
fdelmodc2D_singleGPU: $(MOBJ2) $(OBJC) $(OBJCU)
	nvcc $(LDFLAGS) $(CFLAGS) $(CUFLAGS) -Xcompiler "$(OPTC)" -o $@ $(MOBJ2) $(OBJC) $(LIBS) $(OBJCU)

# # Single-GPU executable (test)
# $(PRG): $(MOBJ) $(OBJC) $(OBJCU)
# 	nvcc $(LDFLAGS) $(CFLAGS) $(CUFLAGS) -Xcompiler "$(OPTC)" -o $@ $(MOBJ) $(OBJC) $(LIBS) $(OBJCU)

old:
	#Old version for reference
	#$(PRG): $(MOBJ) $(OBJC) fdelmodc.h $(OBJCU)
		#nvcc $(LDFLAGS) $(CFLAGS) -Xcompiler "$(OPTC)" -o fdelmodc $(MOBJ) $(OBJC) -Xcompiler \"$(LIBS)\" #did not work
		#nvcc $(LDFLAGS) $(CFLAGS) -Xcompiler "$(OPTC)" -o fdelmodc $(MOBJ) $(OBJC) -L/home/victor.ramalho/REPOSITORIES-Ogun/OpenSource-forked/lib -lgenfft -L/opt/intel/2018/mkl/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -L/opt/intel/2018/mkl/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl $(OBJCU) # works
		#nvcc $(LDFLAGS) $(CFLAGS) -Xcompiler "$(OPTC)" -o fdelmodc $(MOBJ) $(OBJC) $(LIBS) $(OBJCU) #should work if there are no --start-group, -Wl, --end-group on $LIBS
		#$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o fdelmodc $(OBJC) $(LIBS) # original

# Compile main.cu and cuobjs
%.o: %.cu
	nvcc -Xcompiler "$(OPTC)" -c $< 

install: fdelmodc2D_singleGPU
	cp fdelmodc2D_singleGPU $B

clean:
		rm -f core $(OBJC) $(OBJCU) $(MOBJ) $(MOBJ2) fdelmodc2D_singleGPU fdelmodc2D_multicore fdelmodc2D_multiGPU $B/fdelmodc2D*

realclean:
		rm -f core $(OBJC) $(OBJM) $(PRG) $B/fdelmodc 


print:	Makefile $(SRC)
	$(PRINT) $?
	@touch print

count:
	@wc $(SRC)

tar:
	@tar cf $(PRG).tar Makefile $(SRC) && compress $(PRG).tar



