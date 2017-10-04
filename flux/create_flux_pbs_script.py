import numpy as np


def create_flux_pbs_script(python_file, dict_of_parameter_lists, script_file_name='flux_pbs.sh', jobname='flux_job', procs=1, memory_mb=4000, walltime='6:00:00', email='cmiless@umich.edu'):
    """ creates a flux pbs script as output to enclosed folder. It will take in a function and lists of parameters to sweep over as input. """

    # create file
    f = open(script_file_name, 'w')

    # Create PBS preamble and cd commands
    f.write("""# !/bin/sh
#### PBS preamble

#PBS -N %s""" % jobname + """

# Your email:
#PBS -M  %s""" % email + """
#PBS -m a
#PBS -V

# Change the number of cores (ppn=1), amount of memory, and walltime:
#PBS -l procs=%s""" % str(procs) + "," + "pmem=%s" % str(memory_mb) + "mb,walltime=%s" % walltime + """

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

""")

    # Create parameter lists
    for parameter_name, parameter_list in dict_of_parameter_lists.items():
        f.write(str(parameter_name) + "_list=(")
        for parameter in parameter_list:
            f.write(" " + str(parameter))
        f.write(" )\n")

    # Create empty argments list and initialize counter
    f.write("params=()\n")
    f.write("count=1\n")
    f.write("\n")

    # create necessary for-loop statements in script file
    single_indentation = "   "
    for i, (parameter_name, parameter_list) in enumerate(dict_of_parameter_lists.items()):
        indentation = single_indentation * i
        f.write(indentation + "for %s" % parameter_name +
                " in ${" + parameter_name + "_list[@]}\n")
        f.write(indentation + "do\n")

    # Write if statement to match run with flux server queue number
    indentation += single_indentation

    f.write(indentation +
            """if [ "$count" = "$PBS_ARRAYID" ]\n""" + indentation + """then\n""")

    # create description list
    indentation += single_indentation
    f.write(indentation + 'description=("')
    for i, (parameter_name, parameter_list) in enumerate(dict_of_parameter_lists.items()):
        if i > 0:
            f.write("-")
        f.write(parameter_name + "=$" + parameter_name)
    f.write('")\n')

    # create argument list
    f.write(indentation + 'arguments=("')
    for i, (parameter_name, _) in enumerate(dict_of_parameter_lists.items()):
        if i > 0:
            f.write(" ")
        f.write("$" + parameter_name)
    f.write('")\n')

    # Write commands to run python file
    f.write(indentation + "python " + python_file +
            " $arguments -> log-$description.txt")
    f.write("\n")

    # end if statement
    indentation = indentation[: -len(single_indentation)]
    f.write(indentation + "fi")
    f.write("\n")

    # count
    f.write(indentation + "count=`expr $count + 1` \n")

    # end for loop statements
    for i, (parameter_name, parameter_list) in enumerate(dict_of_parameter_lists.items()):
        indentation = indentation[: -len(single_indentation)]
        f.write(indentation + "done\n")

    # Close file
    f.close()


if __name__ == '__main__':
    create_flux_pbs_script('sqrt_func.py',
                           {'Pe': [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0],
                            'L': [1.0, 16.0, 32.0]})
