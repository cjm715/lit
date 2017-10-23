from integrators import RK4_timestepper, mega_RK4_timestepper, integrator2, integrator
from tools import create_grid, dt_cfl
from operators import OperatorKit
import numpy as np


def lit_enstrophy_sim(N, L, Pe, T, reported_num_time_pts=100, gamma=1.0, cfl=True):
    kappa = 1.0 / Pe

    # Create tool box
    okit = OperatorKit(N, L, kappa)
    # Initial condition
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])

    # Create operators: d th / dt = operator (th)
    def lit_enstrophy_op(scalar):
        return okit.lit_enstrophy_op(scalar, gamma, dealias=False)

    def sin_op(scalar):
        return okit.sin_flow_op(scalar)

    time_array = np.linspace(0, T, reported_num_time_pts)

    # perform perturbation
    th0 = RK4_timestepper(sin_op, th0, 0.001 * T)

    if cfl is True:
        dt0_cfl = 0.01 * dt_cfl(N, L, kappa, gamma * L)
        th = integrator2(lit_enstrophy_op, mega_RK4_timestepper,
                         th0, time_array, dt0_cfl)
    else:
        th = integrator(lit_enstrophy_op, RK4_timestepper, th0, time_array)

    return [time_array, th]
