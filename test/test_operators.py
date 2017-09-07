from tools import ScalarTool, create_grid, VectorTool
from operators import OperatorKit
import numpy as np
import pytest


def test_that_adv_diff_operator_output_is_real():
    L = 10.0
    N = 128
    kappa = 0.1
    X = create_grid(N, L)
    u = np.zeros((2, N, N))
    u[0] = np.sin((2.0 * np.pi / L) * X[1])
    th = np.sin((2.0 * np.pi / L) * X[0])
    op = OperatorKit(N, L, kappa)
    assert np.all(np.isreal(op.adv_diff_op(u, th)))


def test_that_average_of_adv_diff_operator_is_0():
    L = 10.0
    N = 128
    kappa = 0.1
    X = create_grid(N, L)
    u = np.zeros((2, N, N))
    u[0] = np.sin((2.0 * np.pi / L) * X[1])
    th = np.sin((2.0 * np.pi / L) * X[0])
    op = OperatorKit(N, L, kappa)
    mean_op = np.mean(op.adv_diff_op(u, th))
    assert np.isclose(mean_op, 0.0)


def test_integral_of_theta_times_adv_diff_op_is_equal_to_neg_kappa_h1norm_sq():
    L = 10.0
    N = 256
    kappa = 0.1
    X = create_grid(N, L)
    st = ScalarTool(N, L)
    u = np.zeros((2, N, N))
    u[0] = np.sin((2.0 * np.pi / L) * X[1])
    th = np.sin((2.0 * np.pi / L) * X[0])
    op = OperatorKit(N, L, kappa)
    assert np.isclose(
        st.sint(th * op.adv_diff_op(u, th)), - kappa * st.h1norm(th)**2.0)


def test_that_adv_diff_operator_output_size():
    L = 10.0
    N = 128
    kappa = 0.1
    X = create_grid(N, L)
    u = np.zeros((2, N, N))
    u[0] = np.sin((2.0 * np.pi / L) * X[1])
    th = np.sin((2.0 * np.pi / L) * X[0])
    op = OperatorKit(N, L, kappa)
    assert np.shape(op.adv_diff_op(u, th)) == (N, N)


def test_that_average_of_adv_diff_operator_hat_is_0():
    L = 10.0
    N = 128
    kappa = 0.1
    X = create_grid(N, L)
    st = ScalarTool(N, L)
    vt = VectorTool(N, L)
    u = np.zeros((2, N, N))
    u[0] = np.sin((2.0 * np.pi / L) * X[1])
    th = np.sin((2.0 * np.pi / L) * X[0])

    u_hat = vt.fft(u)
    th_hat = st.fft(th)

    op = OperatorKit(N, L, kappa)

    mean_op = np.mean(st.ifft(op.adv_diff_op_hat(u_hat, th_hat)))
    assert np.isclose(mean_op, 0.0)


def test_integral_of_theta_times_ifft_adv_diff_operator_hat_is_equal_to_neg_kappa_h1norm_sq():
    L = 10.0
    N = 128
    kappa = 1.0
    X = create_grid(N, L)
    st = ScalarTool(N, L)
    vt = VectorTool(N, L)
    u = np.zeros((2, N, N))

    # u[0] = np.sin((2.0 * np.pi / L) * X[1])
    th = np.sin((2.0 * np.pi / L) * X[0])

    u_hat = vt.fft(u)
    th_hat = st.fft(th)

    op = OperatorKit(N, L, kappa)
    op_output = st.ifft(op.adv_diff_op_hat(u_hat, th_hat))

    assert np.isclose(st.sint(th * op_output), - kappa * st.h1norm(th)**2.0)


def test_that_LIT_energy_flow_satisfies_budget_constriant():
    L = 10.0
    N = 128
    kappa = 0.1
    U = 2.0
    vt = VectorTool(N, L)
    st = ScalarTool(N, L)
    th = np.sin((2 * np.pi / L) * st.X[0])
    okit = OperatorKit(N, L, kappa)
    assert np.isclose(vt.l2norm(okit.u_lit_energy(th, U)), U * L)


def test_that_LIT_energy_flow_is_incompressible():
    L = 10.0
    N = 128
    kappa = 0.1
    U = 2.0
    vt = VectorTool(N, L)
    st = ScalarTool(N, L)
    th = np.sin((2 * np.pi / L) * st.X[0])
    okit = OperatorKit(N, L, kappa)
    assert vt.is_incompressible(okit.u_lit_energy(th, U))


def test_that_LIT_enstrophy_flow_satisfies_budget_constraint():
    L = 10.0
    N = 128
    kappa = 0.1
    gamma = 2.0
    vt = VectorTool(N, L)
    st = ScalarTool(N, L)
    th = np.exp(np.sin((2 * np.pi / L) *
                       st.X[0]) * np.cos((2 * np.pi / L) * st.X[1]))
    okit = OperatorKit(N, L, kappa)
    assert np.isclose(
        st.l2norm(vt.curl(okit.u_lit_enstrophy(th, gamma))), gamma * L)


def test_that_LIT_enstrophy_flow_is_incompressible():
    L = 10.0
    N = 128
    kappa = 0.1
    gamma = 2.0
    vt = VectorTool(N, L)
    st = ScalarTool(N, L)
    th = np.exp(np.sin((2 * np.pi / L) *
                       st.X[0]) * np.cos((2 * np.pi / L) * st.X[1]))
    okit = OperatorKit(N, L, kappa)
    assert vt.is_incompressible(okit.u_lit_enstrophy(th, gamma))