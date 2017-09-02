from integrators import RK4, RK4_timestepper, FE_timestepper
from tools import ScalarTool, VectorTool, create_grid, dt_cfl
from operators import OperatorKit
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    # Parameters
    L = 1.0
    N = 256
    U = 1.0
    T = 2.0
    Pe_list = np.linspace(5, 100, 20)

    for Pe in Pe_list:
        print(Pe)
        kappa = 1.0 / Pe
        # Create tool box
        st = ScalarTool(N, L)
        okit = OperatorKit(N, L, kappa)

        # Initial condition
        X = create_grid(N, L)
        th0 = np.sin((2.0 * np.pi / L) * X[0])
        th0_hat = st.ifft(th0)

        # Create operators: d th / dt = operator (th)
        def lit_energy_op_hat(scalar_hat):
            return okit.lit_energy_op_hat(scalar_hat, U)

        def sin_op_hat(scalar_hat):
            return okit.sin_flow_op_hat(scalar_hat)

        dt = 0.25 * dt_cfl(N, L, kappa, U)
        print('dt= ', dt)
        time = np.linspace(0, T, round(T / dt))

        th0_hat = RK4_timestepper(sin_op_hat, th0_hat, 0.001)
        # th_hist_hat = RK4(lit_energy_op_hat, th0_hat, time)
        th_hist_hat = RK4(lit_energy_op_hat, th0_hat, time)

        output = open('pe=%d' % Pe + '.pkl', 'wb')
        pickle.dump([time, th_hist_hat], output)
        output.close()
