from integrators import RK4, RK4_timestepper, FE_timestepper
from tools import ScalarTool, VectorTool, create_grid, dt_cfl
from post_processing import plot_norms
from operators import OperatorKit
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Parameters
    L = 1.0
    N = 256
    Pe = 100.0
    kappa = 1.0 / Pe
    U = 1.0
    T = 0.2

    # Create tool box
    st = ScalarTool(N, L)
    okit = OperatorKit(N, L, kappa)

    # Initial condition
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])
    th0_hat = st.fft(th0)

    # Create operators: d th / dt = operator (th)
    def lit_energy_op_hat(scalar_hat):
        return okit.lit_energy_op_hat(scalar_hat, U)

    def sin_op_hat(scalar_hat):
        return okit.sin_flow_op_hat(scalar_hat)

    def lit_enstrophy_op_hat(scalar_hat):
        return okit.lit_enstrophy_op_hat(scalar_hat, U)

    dt = 0.25 * dt_cfl(N, L, kappa, U)
    print('dt= ', dt)
    time = np.linspace(0, T, round(T / dt))

    th0_hat = RK4_timestepper(sin_op_hat, th0_hat, 0.001)
    th_hist_hat = RK4(lit_energy_op_hat, th0_hat, time)
    # th_hist_hat = RK4(lit_enstrophy_op_hat, th0_hat, time)
    th2_hist = np.array([st.ifft(th_hat) for th_hat in th_hist_hat])
