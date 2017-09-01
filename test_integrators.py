from operators import OperatorKit
from integrators import FE_timestepper, FE, RK4_timestepper, RK4
import numpy as np
from tools import ScalarTool, create_grid, dt_cfl
from copy import copy


def test_that_FE_timestepper_has_first_order_convergence():
    L = 2.0
    N = 128
    kappa = 0.5
    U = 1.0

    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])
    okit = OperatorKit(N, L, kappa)

    def op(th):
        return okit.sin_flow_op(th)

    st = ScalarTool(N, L)

    # largest time step given by CFL condition
    dt0 = dt_cfl(N, L, kappa, U)
    dt_list = dt0 * 2.0**(-np.arange(0, 3, 1))
    timesteps = 2**(np.arange(0, 3, 1))
    finalth_list = []
    for i, dt in enumerate(dt_list):
        th = copy(th0)
        for step in range(timesteps[i]):
            th = FE_timestepper(op, th, dt)
        finalth_list.append(th)

    error_large_small = abs(
        st.l2norm(finalth_list[0] - finalth_list[2]))
    error_medium_small = abs(
        st.l2norm(finalth_list[1] - finalth_list[2]))

    p = np.log2(error_large_small / error_medium_small - 1)
    print('p = ', p)
    assert abs(p - 1) < 0.1


def test_mean_of_th_is_zero_at_end_time():
    L = 2.0
    N = 128
    kappa = 0.1
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])
    okit = OperatorKit(N, L, kappa)

    def op(th):
        return okit.sin_flow_op(th)

    tarray = np.linspace(0, 0.01, 101)
    th = FE(op, th0, tarray)

    assert np.isclose(np.mean(th[-1]), 0)


def test_that_RK4_timestepper_has_fourth_order_convergence():
    L = 2.0
    N = 128
    kappa = 0.05
    U = 1.0

    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])
    okit = OperatorKit(N, L, kappa)

    def op(th):
        return okit.sin_flow_op(th)

    st = ScalarTool(N, L)

    # largest time step given by CFL condition
    dt0 = dt_cfl(N, L, kappa, U)
    dt_list = dt0 * 2.0**(-np.arange(0, 3, 1))
    timesteps = 2**(np.arange(0, 3, 1))
    finalth_list = []
    for i, dt in enumerate(dt_list):
        th = copy(th0)
        for step in range(timesteps[i]):
            th = RK4_timestepper(op, th, dt)
        finalth_list.append(th)

    error_large_small = abs(
        st.l2norm(finalth_list[0] - finalth_list[2]))
    error_medium_small = abs(
        st.l2norm(finalth_list[1] - finalth_list[2]))

    p = np.log2(error_large_small / error_medium_small - 1)
    print('p = ', p)
    assert abs(p - 4) < 0.1


def test_that_RK4_timestepper_using_hatted_formulation_has_fourth_order_convergence():
    L = 2.0
    N = 128
    kappa = 0.05
    U = 1.0
    st = ScalarTool(N, L)
    X = create_grid(N, L)

    th0_hat = st.fft(np.sin((2.0 * np.pi / L) * X[0]))
    okit = OperatorKit(N, L, kappa)

    def op_hat(th_hat):
        return okit.sin_flow_op_hat(th_hat)

    # largest time step given by CFL condition
    dt0 = dt_cfl(N, L, kappa, U)
    dt_list = dt0 * 2.0**(-np.arange(0, 3, 1))
    timesteps = 2**(np.arange(0, 3, 1))
    finalth_list = []
    for i, dt in enumerate(dt_list):
        th_hat = copy(th0_hat)
        for step in range(timesteps[i]):
            th_hat = RK4_timestepper(op_hat, th_hat, dt)
        finalth_list.append(st.ifft(th_hat))

    error_large_small = abs(
        st.l2norm(finalth_list[0] - finalth_list[2]))
    error_medium_small = abs(
        st.l2norm(finalth_list[1] - finalth_list[2]))

    p = np.log2(error_large_small / error_medium_small - 1)
    print('p = ', p)
    assert abs(p - 4) < 0.1


def test_confirm_equal_hatted_vs_unhatted_formulations_given_the_same_results():
    # Parameters
    L = 2.0
    N = 64
    kappa = 0.1

    # Create tool box
    st = ScalarTool(N, L)
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])
    th0_hat = st.fft(th0)
    okit = OperatorKit(N, L, kappa)

    # Define operator functions
    def op(th):
        return okit.sin_flow_op(th)

    def op_hat(th_hat):
        return okit.sin_flow_op_hat(th_hat)

    # Integrate
    tarray = np.linspace(0, 0.1, 100)
    th_hat = RK4(op_hat, th0_hat, tarray)  # Hatted formulation
    th = np.real(RK4(op, th0, tarray))  # Unhatted formulation
    th2 = np.real(st.ifft(th_hat[-1]))
    assert np.allclose(th2, th[-1])


def test_RK4_output_is_real_when_operator_is_unhatted():
    # Parameters
    L = 2.0
    N = 64
    kappa = 0.1

    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])
    okit = OperatorKit(N, L, kappa)

    # Define operator functions
    def op(th):
        return okit.sin_flow_op(th)

    # Integrate
    tarray = np.linspace(0, 0.1, 100)
    th = RK4(op, th0, tarray)  # Unhatted formulation
    assert np.all(np.isreal(th))


def test_lit_energy_op_and_lit_energy_op_hat_integrations_give_the_same_results():
    # Parameters
    L = 1.0
    N = 64
    Pe = 100.0
    kappa = 1.0 / Pe
    U = 1.0

    # Create tool box
    st = ScalarTool(N, L)
    okit = OperatorKit(N, L, kappa)

    # Initial condition
    X = create_grid(N, L)
    th0 = np.sin((2.0 * np.pi / L) * X[0])
    th0_hat = st.fft(th0)

    # Create operators: d th / dt = operator (th)
    def lit_op_hat(scalar_hat):
        return okit.lit_energy_op_hat(scalar_hat, U)

    def lit_op(scalar):
        return okit.lit_energy_op(scalar, U)

    def sin_op_hat(scalar_hat):
        return okit.sin_flow_op_hat(scalar_hat)

    def sin_op(scalar):
        return okit.sin_flow_op(scalar)

    time = np.linspace(0, 0.02, 20)
    th0 = RK4_timestepper(sin_op, th0, 0.001)
    th_hist = RK4(lit_op, th0, time)

    th0_hat = RK4_timestepper(sin_op_hat, th0_hat, 0.001)
    th_hist_hat = RK4(lit_op_hat, th0_hat, time)
    th2_hist = np.array([st.ifft(th_hat) for th_hat in th_hist_hat])

    print(np.amax(abs(th2_hist - th_hist)))
    assert np.allclose(th2_hist, th_hist)
