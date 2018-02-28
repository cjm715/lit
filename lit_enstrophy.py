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
    # Parameters
    L = 1.0
    Pe = float(sys.argv[1])
    print('Pe = ', Pe)
    kappa = 1.0 / Pe
    gamma = 1.0
    T = 13.0

    lb = (kappa / gamma)**0.5
    l_smallest = 0.2 * lb
    print('lb = ', lb)

    M = L / l_smallest
    print('N_boyd = ', N_boyd(M))
    N = min(N_boyd(M), 512)
    # N = 64
    # N = int(sys.argv[2])
    print('N = ', N)

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
    th0 = RK4_timestepper(sin_op, th0, 0.0001)
    start_time = time.time()
    dt0_cfl = 0.1 * dt_cfl(N, L, kappa, gamma * L)
    th = integrator2(lit_enstrophy_op, mega_RK4_timestepper,
                     th0, time_array, dt0_cfl)

    print(time.time() - start_time)

    # Output
    output_folder = 'output-pe=%d' % Pe + '/'
    os.system('mkdir ' + output_folder)
    os.system('cp lit_enstrophy.py ' + output_folder)

    plt.figure()
    st.plot(th[-1])
    plt.savefig(output_folder + 'plot_final_frame-pe=%d' % Pe + '.png')

    plt.figure()
    plot_norms(time_array, th, N, L)
    plt.savefig(output_folder + 'plot_norms-pe=%d' % Pe + '.png')

    movie(time_array, th, N, L, output_path=output_folder)

    output = open(output_folder + 'pe=%d' % Pe + '.pkl', 'wb')
    pickle.dump([time_array, th], output)
    output.close()
