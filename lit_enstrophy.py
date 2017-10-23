from simulations import lit_enstrophy_sim
from integrators import RK4_timestepper, mega_RK4_timestepper, integrator2
from tools import ScalarTool, create_grid, dt_cfl, N_boyd
from post_processing import plot_norms, movie
from operators import OperatorKit
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import pickle
import os
import sys

if __name__ == "__main__":
    Pe = float(sys.argv[1])
    N = int(float(sys.argv[2]))
    # Parameters
    L = 1.0
    gamma = 1.0
    T = 6.0
    kappa = 1.0 / Pe

    print('Pe = ', Pe)
    print('N = ', N)

    [time_array, th] = lit_enstrophy_sim(N, L, Pe, T, cfl=True)

    # Output
    output_folder = 'output-pe=%d' % Pe + '-N=%d' % N + '/'
    os.system('mkdir ' + output_folder)
    os.system('cp lit_enstrophy.py ' + output_folder)

    plt.figure()
    st = ScalarTool(N, L)
    st.plot(th[-1])
    plt.savefig(output_folder + 'plot_final_frame-pe=%d' %
                Pe + '-N=%d' % N + '.png')

    plt.figure()
    plot_norms(time_array, th, N, L)
    plt.savefig(output_folder + 'plot_norms-pe=%d' % Pe + '-N=%d' % N + '.png')

    movie(time_array, th, N, L, output_path=output_folder)

    output = open(output_folder + 'pe=%d' % Pe + '-N=%d' % N + '.pkl', 'wb')
    pickle.dump([time_array, th], output)
    output.close()
