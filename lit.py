from integrators import RK4, RK4_timestepper, FE_timestepper, mega_RK4_timestepper, integrator2
from tools import ScalarTool, VectorTool, create_grid, dt_cfl, N_boyd
from post_processing import plot_norms, movie
from operators import OperatorKit
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

if __name__ == "__main__":
    # Parameters
    L = 1.0
    Pe = 50.0
    kappa = 1.0 / Pe
    U = 1.0
    T = 1.0

    lb = kappa / U
    l_smallest = lb
    print('lb = ', lb)
    # print('l0 = ', l0)
    # print('l_smallest = ', l_smallest)

    M = L / l_smallest
    N = min(N_boyd(M), 512)
    # N = 128
    print('N = ', N)
    # Create tool box
    okit = OperatorKit(N, L, kappa)
    st = ScalarTool(N, L)

    # Initial condition
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])

    # Create operators: d th / dt = operator (th)
    def lit_energy_op(scalar):
        return okit.lit_energy_op(scalar, U)

    def lit_enstrophy_op(scalar):
        return okit.lit_enstrophy_op(scalar, U)

    def sin_op(scalar):
        return okit.sin_flow_op(scalar)

    time_array = np.linspace(0, T, 100)
    th0 = RK4_timestepper(sin_op, th0, 0.001)
    start_time = time.time()

    dt0_cfl = 0.25 * dt_cfl(N, L, kappa, U)
    th = integrator2(lit_energy_op, mega_RK4_timestepper,
                     th0, time_array, dt0_cfl)

    print(time.time() - start_time)

    output_folder = 'output-pe=%d' % Pe + '/'
    os.system('mkdir ' + output_folder)
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
