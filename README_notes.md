```
```
# fdelmodc2D,3D, gpu, multicore

Onde estamos: testando `fdelmodc2D_singleGPU`. Comentamos a parte do host. Se quisermos fazer analise comparativa, descomentar. Ai 
vao ser gerados snaps e tiros cpu e gpu, pra gente tirar a diferenca.

# Error history

## Erro: boundariesP ld 
```
elastic4dc.c:(.text+0x133b): undefined reference to `boundariesP'
elastic4dc.c:(.text+0x1cfb): undefined reference to `boundariesV'
```
Isso aconteceu porque o make estava compilando `boundaries.c` e gerando `boundaries.o` e o nvcc estava compilando `boundaries.cu` e gerando TAMBÉM 
`boundaries.o`. Por via das dúvidas, então, sempre usar nome diferente para versão CUDA das funções.

# Troubleshooting fdelmodc3d

Goal is to troubleshoot Joeri's 3d version

# makemod 
`makemod sizex=1000 sizez=1000 dx=10.0 dz=10.0 cp0=1900 ro0=1200 orig=-500,0 file_base=hom.su`

sizex, sizez in meters. Since *all distances are 0-based*, creates a (nz,nx)=(101,101) model.

`orig` place the origin for SU-plot, SU-access. The reference is the middle of the model.

`orig=-500,0` places the origin on the middle of the model.

`orig=0,0` places the origin on the top left corner.

# makewave

Use `shift=1` to force wlet to be causal (it can wraparound if t0, which defines the peak position, is not far enough from 0).

# fdelmodc

# Overall staggered grid explanation

Thorbecke allocates a pressure field normally, extended for ABCs and half stencil (ex: 201x201 -> (201+100+4)x(201+100+4) = 305x305). He then sets the upper and lower limits  
for each loop for different variables, vx, vz, P, making the grid staggered. *The only thing that makes the grid staggered are the upper and lower limits of the loops*.  
For instance,  
vx has the limits: `mod.ioXx, mod.ieXx` on x-loop; `mod.ioXz, mod.ieXz` on z-loop;   
vz has the limits: `mod.ioZx, mod.ieZx` on x-loop; `mod.ioZz, mod.ieZz` on z-loop;   
P has the limits: `mod.ioPx, mod.iePx` on x-loop; `mod.ioPz, mod.iePz` on z-loop;   

These limits already include boundaries (são já o recheio do modelo).

# Example 

Ex:

```
../../fdelmodc \
    file_cp=hom_cp.su ischeme=1 iorder=4 \
    file_den=hom_ro.su \
    file_src=wave.su \
    file_rcv=shot_hom_fd.su \
    src_type=1 \
        src_orient=1 \
        src_injectionrate=1 \
    rec_type_vz=0 \
    rec_type_p=1 \
    rec_int_vz=2 \
	dtrcv=0.0005 \
        rec_delay=0.1 \
    verbose=2 \
    tmod=2.0 \
    dxrcv=10.0 \
	xrcv1=-500 xrcv2=500 \
    zrcv1=0 zrcv2=0 \
	xsrc=0 zsrc=0 \
    ntaper=30 \
    left=4 right=4 top=4 bottom=4 \
	sna_type_p=1 tsnap1=0.1 tsnap2=2.0 
```

DO NOT comment a line! It'll stop execution and won't read the variables below.

`dz,dx` come from `velFile`, `dt` from `srcFile`. 

# snaps

To visualize .su file use the shell `sunimage.sh` (on this folder)

# SU header parameters

Information on all possible SU header parameters are found in `segy.h`.

The demo `fdelmodc_plane.scr` calls `makemod` and builds a syncline velfield and rhofield.
`surange <model_cp.su` gives:
```
701 traces:
tracl    0 700 (0 - 700)
tracf    0 700 (0 - 700)
trid     130
scalco   -1000
gx       0 2100000 (0 - 2100000)
ns       701
timbas   25
trwf     701
d1       3.000000
d2       3.000000
```

Analysing `fdelmodc`'s function `getModelInfo`, the only header parameters of velfield and rhofield that are used are:
`trid, gx, ns, d1, d2`. Therefore in our custom 3D model building functions, we will only fill them. List:
- `trid` (int) the trace ID. Should be set to TRID_DEPTH (=130) in order for fdelmodc reading in z-direction (`axis` variable). Obs: axis seems to be an useless variable.
- `gx` (int) is distance from x-origin (usually 0) in meters*1000. This is because gx is integer, and 1000 is to preserve decimals. A code that makes use of `gx` should perform `true_gx=(float) hdr_gx/1000` for correctness.
- `ns` (int) are the samples in the z-direction
- `d1` (float) is sample spacing in meters in the z-direction
- `d2` (float) is sample spacing in meters in the x-direction

A 3D model SU-header will have `trid, gx, ns, d1, d2`, and for the additional y-dimension, `gy`:
- `gy` (int) is distance in meters from y-origin (same remarks as in gx; *1000, etc)

For `fdelmodc`(2D), `nx` is automatically figured with model's archive size (`fileSize`) and `ns`.

For `fdelmodc3D`, we need to find `nx, ny, dy`. They are figured out with the process:
- `ny` is figured first from `gy` and `fileSize`
- `nx` is figured from `fileSize` and `ny`
- `dy` is figured from `gy`




# Functions in the program

## Skeleton

fdelmodc  
   |  
   |                                          
getParameters3D ------- getModelInfo3D (OK), getWaveletInfo3D (OK), recvPar3D (OK')  
   |   
   |  
   ALLOCATION 1 (constants): roz, rox, l2m, and (tss, tep, q, lam, mul, tss, tes, tep, r, p, q)  
   |  
allocStoreSourceOnSurface3D (OK')  
   |  
   |  
readModel3D (OK')-------------writesufile3D, writesufilesnwav3D  
   |  
   |  
   ALLOCATION 2 (source): src_nwav  
   |  
defineSource3D(OK')  
   |  
   |  
   ALLOCATION 3 (fields): vz, vx, tzz (and txz and txx for non-acoustic)  
   ALLOCATION 4 (receiv): can be rec_vz, rec_vx and many others  
   ALLOCATION 5 (beams)  
   |  
   |  
   "Sink" sources and receivers  
   |  
writeSrcRecPos3D  
   |  
   |  
   Set MPI workload division (shots)  
   |  
   |  
   LOOP SHOTS  
   |  
   Initialize wavefields (vz,vx,tzz,etc) to zero before every shot  
   |  
   |  
   LOOP TIME  
   |  
   Open OpenMP parallel region  
   |  
acoustic4_3D---------------------boundariesP3D(OK), applySource3D(OK'), storeSourceOnSurface3D(OK'), boundariesV3D(OK'), restoreSourceOnsurface3D(OK')  
   |  
   |  
After timestep part 1(getRecTimes, writeSnapTimes, getBeamTimes)
   |  
After timestep part 2(estimate shot time, output to user)
   |  
   |  
   Close OpenMP parallel region  
   |  
   END LOOP TIME  
   |  
After full modeling part 3 (scale rec's amplitudes)
   |  
After full modeling part 4 (writeRec, writeBeams)   
   |  
   END LOOP SHOTS  
   |  
   Output total computed time  
   |  
   |  
   Frees rox,roz, l2m, src_nwav, vx, vz, tzz  
   |  
freeStoreSourceOnSurface  
   |  
   Frees all remaining variables  
   |  
   |  
   MPI finalize  
   |  
   return 0   
   
# getParameters3D

Part0: get filenames for modPar, wavPar and some general parameters  

Part1: fills part of modPar (subfunctions: getModelInfo)  

Part2: fills wavPar, part of modPar (subfunctions: getWaveletInfo)  

Part3: mal-colocada; fills part of srcPar and modPar  

Part4: stab/disp analysis  

Part5: fills bndPar

Part6: fills modPar limits

Part7 (desorganizado): fills bnd.surface, prints boundary info

Part8: fills shotPar, part of srcPar

Part9 (BIG): random/plane wavPar, srcPar

Part10: src.n and output wlet info

Part11: fills snaPar and beams

Part12: fills recPar and output rec info (subfunctions: recvPar)

## (OK) getParameters3D Part 0

Get filenames for modPar, wavPar and some general parameters  

General Parameters: verbose, disable_check  
modPar: iorder, ischeme, sh, file_cp, file_rho, file_cs, grid_dir  
wavPar: file_src
snaPar: file_snap, file_beam
srcPar: src_at_rcv

## (OK) getParameters3D Part 1: fill modPar (partially) through getModelInfo3D

Get nz, nx, dz, dx, maxvel and minvel for velfield and rhofield. 

### (OK) getModelInfo3D
   Reading all 3D parameters correctly. The axis variable print says "sample represent z-axis".
   Get z,x,y origin coordinates from model headers. For our tests, (f1,f2,f3)=(0,0,0)
   (f1,f2,f3) are saved as (sub_z0,sub_x0,sub_y0) in *getModelParameters3D*.

## (OK) getParameters3D Part 2: fills wavPar, part of modPar, part of recPar

Fills wavPar, part of modPar

wavPar filled: ns, nx, ds, nt, dt, fmax
modPar filled: tmod, dt, nt
recPar filled: delay

Diff 2d and 3d: only changed int to long.
getWaveletInfo3D fill wavPar variables: ns, nx, ds. It can also use wavPar variables to fill: d2, fmax.
If fmax is not provided it finds it by FFTing the wlet, finding max amplitude and assumes fmax corresponding to the amplitude 400 times weaker.
It also interpolates wavelet's dt to model's dt (if different).
It also checks source type and orientation.
Finally we decide to end part2 after determination of disp and stab factors.

wavPar e srcPar eh uma bagunca.

## getParameters3D Part 3: fills (part of) srcPar, modPar

Checks if src_orient is coherent with modeling scheme. Should be inside Part0, IMO.

srcPar: type, orient
modPar: file_qp, file_qs, Qp, Qs, fw, qr

## getParameters3D Part4: stab/disp analysis; fills part of modPar

Sets dispfactor and stabfactor for orders 2 or 4.  
Prints conditions to the user, quits program if criteria not met (unless disable_check==1)


## getParameters3D Part5: fills bndPar

bndPar: lef, rig, top, bot, ntap, npml, R, m, tapx, tapz, tapxz


## getParameters3D Part6: fills modPar limits

modPar: ioXx, ieXx, ioXz, ieXz, ioZx, ieZx, ioZz, ieZz, ioPx, iePx, ioPz, iePz, ioTx, ieTx, ioTz, ieTz  
Obs, remember X and Z don't include boundaries, but P does (cause taper is applied only to Vx and Vz). For PML, where it is 
applied for P, an adjustment has to be made.

## getParameters3D Part7 (desorganizado): fills bnd.surface, prints boundary info

Fills bnd.surface, prints boundary info. 
Parece desorganizado mas eh aparementente necessario nessa ordem porque bnd.surface pode depender de mod.ioPz, que eh 
preenchido na Part6.

## getParameters3D Part8: fills shotPar, part of srcPar

shotPar: n, x[], z[]
srcPar: dip, strike
Obs, checks for source positions in a txt  
Obs, fills source in a circle if "rsrc" is defined  

## getParameters3D Part9 (BIG): random/plane wavPar, srcPar

Fills a bunch of things related to random/plane wavPar, srcPar

## getParameters3D Part10: src.n and output wlet info

Defines src->n and outputs wlet info

## getParameters3D Part11: fills snaPar and beams

fills snaPar and beams

## getParameters3D Part12: fills recPar and output rec info (subfunctions: recvPar)

recPar: sinkdepth, sinkvel, skipdt, nt, int_p, int_vx, int_vz, max_nrec, scale

### recvPar (GRANDE)

recPar: max_nrec, x[], z[], xr[], zr[]

## getParameters3D Part12: fills recPar and output rec info (subfunctions: recvPar)

recPar: type.vz, type.vx, type.ud, type.txx, type.tzz, type.txz, type.pp, type.ss, type.p  
Finally, outputs info to user. End.

# ALLOCATION 1

Todo

# allocStoreSourceOnSurface3D

TODO

# readModel3D 

Todo

# ALLOCATION 2 (source): src_nwav  

The bit of code:
```c
nsamp = 0;
for (i=0; i<wav.nx; i++) {
	src_nwav[i] = (float *)(src_nwav[0] + nsamp);
	nsamp += wav.nsamp[i];
}
```
is copy-paster from SU's ealloc, and serves to allocate contiguously a 2D array (the end of a column is 1-stride address away from the beginning 
of the next column).
Obs, `nst` is the total number of samples in all wavelets; when there are random wavelets, they also have different max sample sizes. Each wlet 
ns is save onto `wav.nsamp[i]`. For instance, 10 random wavelets gives:
```sh
wav.nst=7410
wav.nsamp[0]=1169	nsamp=1169
wav.nsamp[1]=797	nsamp=1966
wav.nsamp[2]=1131	nsamp=3097
wav.nsamp[3]=91	    nsamp=3188
wav.nsamp[4]=844	nsamp=4032
wav.nsamp[5]=489	nsamp=4521
wav.nsamp[6]=1059	nsamp=5580
wav.nsamp[7]=817	nsamp=6397
wav.nsamp[8]=845	nsamp=7242
wav.nsamp[9]=168	nsamp=7410
```
Obs, `src_nwav` for random wavelets occupies `wav.nst*sizeof(float)` bytes, but when it is saved with the `writesufilesrcnwav` (inside `defineSource` function), 
it is saved on top of fixed zeroed traces of size `wav.nt`.


# defineSource

With `srcPar src` parameters already defined, it fills the 2D array `src_nwav` with the wavelet(s). The goal of this function is basically to 
fill this src_nwav array.

# acoustic4_3D

For now, this section will regard notes on acoustic4 (the 2D version).

The acoustic extrapolation does, in order:  
1. Extrap vx for the main field (no boundaries). BODY OF acoustic4. Obs, vx is calculated from p-values. 

2. Extrap vz for the main field (no boundaries). BODY OF acoustic4. Obs, vz is calculated from p-values.  

3. Apply taper/pml to vz, vx. Function: boundariesP. Obs: although it updates vz, vx, it uses the VALUES of P, hence the name.  

4. (RARELY APPLIED) Apply source if source is of type vz,vx (force source)    

4.5 adjust p origins in case of PML (ioPx, ioPz, iePx, iePz)--NOT OUR CASE    

5. Extrapolate p on main field (no boundaries). BODY OF acoustic4.  

5.5 readjust p origins in case of PML (ioPx, ioPz, iePx, iePz)--NOT OUR CASE    

6. Add the src amplitude at position izsrc, ixsrc. Function: applySource  

To better understand the staggered grid, put these prints after getParameters on main function:
```c
```

# applySource

```
int applySource(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float **src_nwav, int verbose)
```
Applies a source (or many sources) to the model. In the GPU version, kernel is called using 1 thread `<<<1,1>>>`

## boundaries

For a simple example of a homogeneous model 201x201. bnd=50. Full dimensions are (201+50+50+4)x(201+50+50+4) = 305x305.  
Using TAPER, normal propagation has the limits
```sh
mod.ioXx = 52 	mod.ieXx = 253
mod.ioZx = 51 	mod.ieZx = 252
mod.ioPx = 1 	mod.iePx = 302
```
So normally P lims grabs the whole grid.  
The TAPER will deal with Vx (i*X*) and Vz(i*Z*) boundaries, while P boundaries *have no taper*.

Using PML, normal propagation has the limits:
```sh
mod.ioXx = 52 	mod.ieXx = 253
mod.ioZx = 51 	mod.ieZx = 252
mod.ioPx = 51 	mod.iePx = 252
```
PML will deal with Vx, Vz (inside boundariesP function) and P boundaries (inside boundariesV function).

CONVENCIONA-SE QUE OS INDICES EM P SAO COMPLETOS. Ajustes de borda sao feitos dentro do modulo boundariesPML
Ajuste deve ser feito na propagacao de P com PML, que ja fizemos
TUdo se resolve com as funcoes como argumento ixo, ixe, izo, ize

### boundariesP
First, if applicable, apply free surface (=1) to ACOUSTIC Vz at the top;  
Seconda, if applicable, apply rigid (=3) boundary conditions;
Third, if applicable, apply PML to Vx, Vz  
Fourth, if applicable, apply taper to Vx, Vz acoustic scheme (used tzz only)
Fifth, if applicable, apply taper to Vx, Vz elastic scheme (uses tzz, txx, txz)

### boundariesV

First, if applicable, apply PML (=2) to acoustic P field (toptop, leflef, toplef, rigrig, rigtop, botbot, botrig, botlef)  
Second, if applicable, applies free surface (=1) to acoustic TZZ (P) field (top, rig, bot, lef)  
Third, if applicable, applies free surface (=1) to ELASTIC Tzz, Txz, Txx fields (top, rig, bot, lef)  

GPU implementation only adapted for First and Second parts.  

# After timestep part 1(getRecTimes, writeSnapTimes, getBeamTimes) - only master thread

If **it** is a multiple of seismogram's dt (skipdt), save surface slice for seismogram through **getRecTimes**  
Then there's other if block with **writeRecTimes** but it seem to never pass, so ignore it.  
Then there is an if block to **writeSnapTimes**. It always passes; selection for coarse dt's is done inside the function.  
Then in the end there is an if block for **getBeamTimes**, which we will ignore for now. 

## getRecTimes

Registering best interpolations based on CPU-GPU difference:  
P:  
intp=0, 1e3 (data) -> 1e-3 (dif)  
intp=1, 1e3 (data) -> 1e2 (dif)  
intp=2, 1e2 (data) -> 1e2 (dif)  
intp=3, 1e3 (data) -> 1e-3 (dif)  

vz: 
intp=3, intvz012, 1e-6 -> 1e-9  
intp=3, intvx012, 1e-4 -> 1e-9  

intp!=3, intvzvx=0, 1e-4 -> 1e-9  
intp!=3, intvzvx=1, 1e-4 -> 1e-9  
intp!=3, intvzvx=2, 1e-4 -> 1e-5  

**Summary**: usar intp=3 (interp bilinear) e intvzvx= qualquer um eh o mais estavel.

# After timestep part 2(estimate shot time, output to user) - only master thread

Pretty straightforward, estimates shot time beforehand

# After full modeling part 3 (scale rec's amplitudes)

Scale rec_p's amplitudes (they were recorded inside previous loop with getRecTimes)

# After full modeling part 4 (writeRec, writeBeams) 

Finally, writes out seismogram for current shot and beams(if applicable)


