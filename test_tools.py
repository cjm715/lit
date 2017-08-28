from tools import ScalarTool
import numpy as np
import math


def test_that_l2norm_of_sinx_on_domain_with_L_of_2pi_equals_sqrt_of_half_of_Lsq():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(2.0 * np.pi * X[0] / L)
    assert math.isclose(st.l2norm(th), (0.5 * L ** 2)**0.5)


def test_hm1norm_of_sinkx_is_sqrt_of_half_of_Lsq_divided_by_k():
    L = 2.0 * np.pi
    N = 128
    k = 10.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[0])
    st = ScalarTool(N, L)

    assert math.isclose(st.hm1norm(th), (0.5 * L ** 2)**0.5 / k)


def test_h1norm_of_sinkx_is_sqrt_of_half_of_Lsq_multiplied_by_k():
    L = 2.0 * np.pi
    N = 128
    k = 10.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[0])
    st = ScalarTool(N, L)

    assert math.isclose(st.h1norm(th), (0.5 * L ** 2)**0.5 * k)


def test_that_sinx_on_domain_with_L_of_2pi_has_equal_norms():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(2.0 * np.pi * X[0] / L)
    l2norm = st.l2norm(th)
    h1norm = st.h1norm(th)
    hm1norm = st.hm1norm(th)
    assert (math.isclose(l2norm, h1norm) and math.isclose(l2norm, hm1norm))


def test_l2norm_of_a_scalar_should_be_nearly_invariant_of_discretization():
    L = 10.0
    Narray = np.array([64, 128, 256, 512]).astype('int')
    l2norm = np.zeros(len(Narray))
    for i, N in enumerate(Narray):
        st = ScalarTool(N, L)
        X = np.mgrid[:N, :N].astype(float) * (L / N)
        th = np.sin(2.0 * np.pi * X[0] / L)
        l2norm[i] = st.l2norm(th)

    assert np.allclose(l2norm[-1], l2norm)


def test_h1norm_of_a_scalar_should_be_nearly_invariant_of_discretization():
    L = 10.0
    Narray = np.array([64, 128, 256, 512]).astype('int')
    h1norm = np.zeros(len(Narray))
    for i, N in enumerate(Narray):
        st = ScalarTool(N, L)
        X = np.mgrid[:N, :N].astype(float) * (L / N)
        th = np.sin(2.0 * np.pi * X[0] / L)
        h1norm[i] = st.h1norm(th)

    assert np.allclose(h1norm[-1], h1norm)


def test_hm1norm_of_a_scalar_should_be_nearly_invariant_of_discretization():
    L = 10.0
    Narray = np.array([64, 128, 256, 512]).astype('int')
    hm1norm = np.zeros(len(Narray))
    for i, N in enumerate(Narray):
        st = ScalarTool(N, L)
        X = np.mgrid[:N, :N].astype(float) * (L / N)
        th = np.sin(2.0 * np.pi * X[0] / L)
        hm1norm[i] = st.hm1norm(th)

    assert np.allclose(hm1norm[-1], hm1norm)


def test_xderivative_of_sinkx_is_kcoskx():
    L = 10.5
    N = 128
    k = 2.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[0])

    st = ScalarTool(N, L)
    gradth = st.grad(th)

    assert np.allclose(gradth[0], k * np.cos(k * X[0]))


def test_yderivative_of_sinky_is_kcosky():
    L = 10.5
    N = 128
    k = 2.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[1])

    st = ScalarTool(N, L)
    gradth = st.grad(th)

    assert np.allclose(gradth[1], k * np.cos(k * X[1]))


def test_laplacian_of_sinky_is_neq_ksq_times_sinky():
    L = 10.5
    N = 128
    k = 2.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[1])

    st = ScalarTool(N, L)
    lapth = st.lap(th)

    assert np.allclose(lapth, -k**2 * np.sin(k * X[1]))


def test_laplacian_of_sinkx_is_neq_ksq_times_sinkx():
    L = 10.5
    N = 128
    k = 2.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[0])

    st = ScalarTool(N, L)
    lapth = st.lap(th)

    assert np.allclose(lapth, -k**2 * np.sin(k * X[0]))
