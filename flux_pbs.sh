# !/bin/sh
#### PBS preamble

#PBS -N lit_enstrophy

# Your email:
#PBS -M  cmiless@umich.edu
#PBS -m a
#PBS -V

# Change the number of cores (ppn=1), amount of memory, and walltime:
#PBS -l procs=1,pmem=8000mb,walltime=10:00:00

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

description=("sim_number = $PBS_ARRAYID")
echo $description

Pe_list=(128.0 256.0 512.0 1024.0 2048.0 4096.0 8192.0 16384.0)
params=()
count=1

for Pe in ${Pe_list[@]}
do
      if [ "$count" = "$PBS_ARRAYID" ]
      then
         description=("Pe=$Pe")
         arguments=("$Pe ")
         python lit_enstrophy.py $arguments -> log-$description.txt
      fi
      count=`expr $count + 1`
done
