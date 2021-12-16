# Modeling GPU Leon Diekmann

Modelagem multi-GPU, divisão por datum.
Recomenda-se uso do taper (10x mais rapido que CPU), já que usar PML
é apenas 20% mais rápido que a CPU.
Feito em 20211215, modelando Greens a cada datum z=fixo para Daniel.
Usa `fdelmodc2D_gpu` e `gnu_parallel`. 
Divide sempre datums completos para cada nó, de forma que cada GPU sempre 
realiza trabalho dentro de um mesmo datum, gerando um arquivo resultante. 
Assim, no `fdelmodc2D_gpu`, `dxshot=etc` e `dzshot=0.0` sempre.

Por padrão as shells estão configuradas para rodar 
```sh
zfoc0=100.0; zfoc1=600.0; dzfoc=100.0
xfoc0=-808.0; xfoc1=808.0; dxfoc=404.0
```
ou seja, 6 datums com 4 pontos cada. O script em GPU utiliza 1 nó com 4
GPUs.

# Requer
- Modelo de velocidades e densidade (`modelLeon_cp.su`, `modelLeon_ro.su`)
- OpenSource JT
- fdelmodc2D_gpu
- gnu_parallel

# Passo 1
```sh
cd demo
cd modeling_leon_diekmann
cp modelLeon_cp.velsu modelLeon_cp.su
cp modelLeon_ro.velsu modelLeon_ro.su
mkdir -p jobs greens
```

# Passo 2

Configurar `green_mgpuGEN.scr` para area de modelagem (zfoc0,zfoc1,etc), wavelet, receptores, dt, etc.
Configurar numero de nós no SBATCH --array.

# Passo 3
Rodar:
```sh
rm -r greens/* jobs/*; sbatch green_mgpuGEN.scr
```

Se quiser fazer um dry run, descomentar loop em inode e colocar um exit após o 
loop. Analisar os echos e arquivos de configuração na pasta `jobs`.

# Passo 4
Concatenar shots para cada datum
```sh
cd greens
for i in cat*; do echo "Doing $i ..."; bash $i; echo "Done"; done
```
Organiza originais e datums:
```sh
mkdir -p orig_greens
mv *gpu_rp.su cat* orig_greens
mkdir datums_greens
mv *.su datums_greens
```

# Passo 5
Deconcatenar todos os datums em arquivos individuais. Nessa pasta os códigos 
utilizam multi-threading com OpenMP para acelerar a separação, visto que 
rodar vários `suwind` sequencialmente pode levar mais tempo que a própria 
modelagem.

Setup:
```sh
cd individual_greens
make
mkdir -p files_su
module load openmpi/4.0.4-gcc
```

Configura `roda.sh`, seleciona a particao. 

Configura `input.dat` com os mesmos parametros `xfoc0,xfoc1,dxfoc`, 
`zfoc0,zfoc1,dzfoc` especificados em `green_mgpuGEN.scr`.

Roda:
```sh
bash submit.sh
```

Os arquivos individuais estarão na pasta `files_su`.
