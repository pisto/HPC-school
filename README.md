Needs boost static libraries, glibc >= 2.22, recent gcc, cmake. All stuff you cannot find in CINECA: compile locally then upload binaries/libs to Galileo:
```
mkdir build
cd build
make -DCMAKE_BUILD_TYPE=Release -DMPI_CXX_COMPILER=/upath/to/mpicxx -DMPI_C_COMPILER=/upath/to/mpicc ..
```
This set of libs should be sufficient:
```
[a08trb21@node166 ~]$ ls -lath .local/lib64/
totale 720K
-rwxr-xr-x 1 a08trb21 corsi 177K 17 mag 14.47 libmvec.so.1
drwxr-xr-x 2 a08trb21 corsi  512 17 mag 14.47 .
lrwxrwxrwx 1 a08trb21 corsi   21 16 mag 10.36 libmpi_cxx.so.20 -> libmpi_cxx.so.20.10.0
-rwxr-xr-x 1 a08trb21 corsi 110K 16 mag 10.36 libmpi_cxx.so.20.10.0
drwxr-xr-x 5 a08trb21 corsi  512 15 mag 16.28 ..
```
and I run commands with `srun {slurm args...} ./mpienv ./mpibologna <program> {args...}`, (find `mpienv` in this repository). For example:
```
[a08trb21@node166 ~]$ srun --mpi=pmi2 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=30G --time=01:00:00 --partition=gll_usr_prod --account=train_scB2018 ./mpienv ./mpibologna add_vector < testvector 
sum(b) 5e+07 for N elements 33331562 in 30ms
```
