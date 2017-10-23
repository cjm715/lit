import numpy as np
import os
import sys


def create_flux_pbs_script(path_to_place_pbs_script, script_file_name, python_file, dict_of_parameter_lists, jobname='flux_job', procs=1, memory_mb=4000, walltime='6:00:00', email='cmiless@umich.edu'):
    """ creates a flux pbs script as output to enclosed folder. It will take in a function and lists of parameters to sweep over as input. """

    # create file
    f = open(path_to_place_pbs_script + script_file_name, 'w')

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


def run_python_file(python_file, dict_of_parameter_lists, flux_folder_name, jobname='flux_job', procs=1, memory_mb=4000, walltime='6:00:00', email='cmiless@umich.edu'):
    """ script will generate a folder with all necessary files to run on the remote server flux
     and excute the script remotely. """

    # make a folder to be uploaded
    os.system("mkdir " + flux_folder_name)

    # add all necessary python files
    os.system("cp " + python_file + " " + flux_folder_name)
    os.system("cp operators.py " + flux_folder_name)
    os.system("cp post_processing.py " + flux_folder_name)
    os.system("cp sol_checking.py " + flux_folder_name)
    os.system("cp tools.py " + flux_folder_name)
    os.system("cp integrators.py " + flux_folder_name)
    os.system("cp operators.py " + flux_folder_name)

    # generate and add pbs script
    pbs_script_file_name = 'flux_pbs.sh'
    create_flux_pbs_script(flux_folder_name, pbs_script_file_name,
                           python_file, dict_of_parameter_lists, jobname=jobname, procs=procs, memory_mb=memory_mb, walltime=walltime, email=email)

    # upload folder with scp
    os.system(
        'sshpass -p "!ch@ot1c!" scp -r ' + flux_folder_name + ' flux-xfer.arc-ts.umich.edu:/home/cmiless')

    # erase folder locally
    os.system("rm -r " + flux_folder_name)

    # determine number of runs
    num_runs = 1
    for _, parameter_list in dict_of_parameter_lists.items():
        num_runs *= len(parameter_list)

    # create login shell script
    ssh_login_script_file_name = 'ssh_login_script.sh'
    f = open(ssh_login_script_file_name, 'w')
    f.write("""#!/usr/bin/expect -f
# ssh into flux
spawn ssh cmiless@flux-login.arc-ts.umich.edu
expect "Password: "
send "!ch@ot1c!\r"
expect "Passcode or option (1-3): "
send "1\r"
expect "]$"

"""
            + 'send "cd %s' % flux_folder_name + '\r"' + '\n'

            + 'send "qsub -t 1-%d' % num_runs + ' ' + pbs_script_file_name + '\r"'
            """

interact

    """)

    f.close()

    # give permission to, run, and delete login shell script
    os.system("chmod 0755 " + ssh_login_script_file_name)
    os.system("./" + ssh_login_script_file_name)
    os.system("rm " + ssh_login_script_file_name)
