# !/bin/sh
#### PBS preamble

#PBS -N lit_enstrophy_inf

# Your email:
#PBS -M  cmiless@umich.edu
#PBS -m a
#PBS -V

# Change the number of cores (ppn=1), amount of memory, and walltime:
#PBS -l procs=4,pmem=8000mb,walltime=24:00:00

# Change "example_flux" to the name of your Flux allocation:
#PBS -A lsa_flux
#PBS -q flux
#PBS -l qos=flux

# End PBS preamble


#  Show list of CPUs you ran on, if you're running under PBS
if [ -n "$PBS_NODEFILE" ]; then cat $PBS_NODEFILE; fi

#  Change to the directory you submitted from
if [ -n "$PBS_O_WORKDIR" ]; then cd $PBS_O_WORKDIR; fi

#  Put your job commands here:

python lit_enstrophy_inf.py
