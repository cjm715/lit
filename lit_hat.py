from integrators import RK4, RK4_timestepper, FE_timestepper
from tools import ScalarTool, VectorTool, create_grid, dt_cfl
from post_processing import plot_norms
from operators import OperatorKit
import numpy as np
import matplotlib.pyplot as plt
import time
from pyfftw.interfaces import cache
from numba import jit
import numba
import pyfftw

if __name__ == "__main__":
    cache.enable()

    # Parameters
    L = 1.0
    N = 128
    Pe = 100.0
    kappa = 1.0 / Pe
    U = 1.0
    T = 0.2

    # Create tool box
    st = ScalarTool(N, L)
    vt = VectorTool(N, L)
    okit = OperatorKit(N, L, kappa)

    # Initial condition
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])
    th0_hat = st.fft(th0)
    product = pyfftw.empty_aligned((N, N), dtype=float)
    product_hat = pyfftw.empty_aligned((N, N // 2 + 1), dtype=complex)
    th = pyfftw.empty_aligned((N, N), dtype=float)
    grad_invlap_th = pyfftw.empty_aligned(
        (2, N, N), dtype=float)
    v = pyfftw.empty_aligned((2, N, N), dtype=float)
    grad_th = pyfftw.empty_aligned((2, N, N), dtype=float)
    out_hat = pyfftw.empty_aligned((N, N), dtype=complex)

    # Create operators: d th / dt = operator (th)
    def lit_energy_op_hat(scalar_hat):
        return okit.lit_energy_op_hat(scalar_hat, U)

    def sin_op_hat(scalar_hat):
        return okit.sin_flow_op_hat(scalar_hat)

    def lit_enstrophy_op_hat(scalar_hat):
        return okit.lit_enstrophy_op_hat(scalar_hat, U)

    dt = 0.25 * dt_cfl(N, L, kappa, U)
    print('dt= ', dt)
    time_array = np.linspace(0, T, round(T / dt))

    th0_hat = RK4_timestepper(sin_op_hat, th0_hat, 0.001)

    start_time = time.time()
    th_hist_hat = RK4(lit_energy_op_hat, th0_hat, time_array)
    print(time.time() - start_time)
    # th_hist_hat = RK4(lit_enstrophy_op_hat, th0_hat, time)
    # th2_hist = np.array([st.ifft(th_hat) for th_hat in th_hist_hat])
    # print
    # plot_norms(time, th2_hist, N, L)
    # plt.savefig('plot.png')
    st.plot(st.ifft(th_hist_hat[-1]))
    plt.savefig('plot.png')
