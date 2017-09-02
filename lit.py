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

    # Create tool box
    okit = OperatorKit(N, L, kappa)

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

    time = np.linspace(0, 0.2, 200)
    th0 = RK4_timestepper(sin_op, th0, 0.001)
    th = RK4(lit_energy_op, th0, time)
    # th = RK4(lit_enstrophy_op, th0, time)
    plot_norms(time, th, N, L)
    plt.savefig('plot.png')
