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
    # Parameters
    L = 1.0
    gamma = 1.0

    Pe = float(sys.argv[1])
    print('Pe = ', Pe)
    kappa = 1.0 / Pe
    T = 6.0

    N = int(float(sys.argv[2]))

    print('N = ', N)

    dt0_cfl = 0.25 * dt_cfl(N, L, kappa, gamma * L)

    # Create tool box
    okit = OperatorKit(N, L, kappa)
    st = ScalarTool(N, L)

    # Initial condition
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])

    # Create operators: d th / dt = operator (th)
    def lit_enstrophy_op(scalar):
        return okit.lit_enstrophy_op(scalar, gamma)

    def sin_op(scalar):
        return okit.sin_flow_op(scalar)

    # Perform simulation
    time_array = np.linspace(0, T, 200)

    th0 = RK4_timestepper(sin_op, th0, dt0_cfl)
    start_time = time.time()

    th = integrator2(lit_enstrophy_op, mega_RK4_timestepper,
                     th0, time_array, dt0_cfl)

    print(time.time() - start_time)

    # Output
    output_folder = 'output-pe=%d' % Pe + '-N=%d' % N + '/'
    os.system('mkdir ' + output_folder)
    os.system('cp lit_enstrophy.py ' + output_folder)

    plt.figure()
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
