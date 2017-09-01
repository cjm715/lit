from tools import ScalarTool, create_grid, VectorTool
from operators import OperatorKit
from post_processing import compute_norms
from integrators import RK4, RK4_timestepper
import numpy as np


def test_l2norm_decay_h1norm_relation():

    # Parameters
    L = 1.0
    N = 64
    Pe = 1000.0
    kappa = 1.0 / Pe
    U = 1.0

    # Create tool box
    okit = OperatorKit(N, L, kappa)

    # Initial condition
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])

    # Create operators: d th / dt = operator (th)
    def lit_op(scalar):
        return okit.lit_energy_op(scalar, U)

    def sin_op(scalar):
        return okit.sin_flow_op(scalar)

    time = np.linspace(0, 0.01, 100)
    th0 = RK4_timestepper(sin_op, th0, 0.001)
    th_hist = RK4(lit_op, th0, time)
    # print(np.shape(th_hist))
    hm1norm_hist, l2norm_hist, h1norm_hist = compute_norms(th_hist, N, L)
    num_pts = len(time)
    l2decay = np.zeros(num_pts - 1)
    h1norm_dependence = -2.0 * kappa * h1norm_hist[:-1]**2

    # print(np.shape(hm1norm_hist))

    for i in range(num_pts - 1):
        dt = time[i + 1] - time[i]
        l2decay[i] = (l2norm_hist[i + 1]**2 - l2norm_hist[i]**2) / dt

    assert np.allclose(h1norm_dependence, l2decay, rtol=0.01)
