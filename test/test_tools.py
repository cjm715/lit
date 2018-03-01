from tools import ScalarTool, VectorTool
import numpy as np
import math
from tools import N_boyd


def test_dealias_of_sinkx_where_k_is_above_two_thirds_dealias_boundary_is_zero():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)

    kabove = np.ceil(st.kmax_dealias) + 1
    th = np.sin(2. * np.pi / L * kabove * st.X[0])
    z = np.zeros(th.shape)

    assert np.allclose(st.dealias(th), z)


def test_dealias_of_sinkx_where_k_is_above_two_thirds_dealias_boundary_is_zero_y_direction():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)

    kabove = np.ceil(st.kmax_dealias) + 1
    th = np.sin(2. * np.pi / L * kabove * st.X[1])
    z = np.zeros(th.shape)

    assert np.allclose(st.dealias(th), z)


def test_dealias_of_sinkx_where_k_is_below_two_thirds_dealias_boundary_is_zero():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)

    k = np.floor(st.kmax_dealias) - 1
    th = np.sin(2. * np.pi / L * k * st.X[0])
    z = np.zeros(th.shape)

    assert not np.allclose(st.dealias(th), z)


def test_dealias_of_sinkx_where_k_is_below_two_thirds_dealias_boundary_is_zero_y_direction():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)

    k = np.floor(st.kmax_dealias) - 1
    th = np.sin(2. * np.pi / L * k * st.X[1])
    z = np.zeros(th.shape)

    assert not np.allclose(st.dealias(th), z)


def test_lap_invlap_of_th_is_th_minus_mean_value():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)

    th = np.random.random((N, N))

    assert np.allclose(st.lap(st.invlap(th)), st.subtract_mean(th))


def test_spatial_integral_of_neg_invlap_th_time_th_equals_hm1_norm_sq():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)

    th = np.random.random((N, N))
    th = st.dealias(th)
    a = st.sint(-1.0 * st.invlap(th) * th)
    b = st.hm1norm(th)**2.
    print(a, b)
    assert np.allclose(a, b)


def test_h1normsq_of_vector_is_spatial_integral_neg_lapvec_times_vec():
    N = 128
    L = 2.0
    kappa = 0.0
    gamma = 1.0
    v = np.random.random((2, N, N))
    st = ScalarTool(N, L)
    vt = VectorTool(N, L)
    v = vt.div_free_proj(v)
    v = vt.dealias(v)

    a = vt.h1norm(v)**2
    b = st.sint(np.sum(-vt.lap(v) * v, 0))
    assert np.allclose(a, b)


def test_h1norm_of_sin_flow():
    N = 128
    L = 2.0
    vt = VectorTool(N, L)
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    u = np.zeros((2, N, N))
    u[0] = np.sin(2. * np.pi / L * X[1])
    assert np.isclose(vt.h1norm(u)**2., 2. * np.pi**2.)


def test_curl_is_equal_to_curl_computed_with_grad_function():
    N = 128
    L = 2.0
    kappa = 0.0
    gamma = 1.0
    v = np.random.random((2, N, N))
    # print(np.shape(v))
    st = ScalarTool(N, L)
    vt = VectorTool(N, L)
    v = vt.div_free_proj(v)
    v = vt.dealias(v)
    c = st.grad(v[1])[0] - st.grad(v[0])[1]
    assert np.allclose(c, vt.curl(v))


def test_l2norm_squared_of_curl_of_vector_is_spatial_integral_of_neg_vector_times_lap_vector():
    N = 128
    L = 2.0
    kappa = 0.0
    gamma = 1.0
    v = np.random.random((2, N, N))
    # print(np.shape(v))
    st = ScalarTool(N, L)
    vt = VectorTool(N, L)
    v = vt.div_free_proj(v)
    v = vt.dealias(v)
    curl = vt.curl(v)
    a = st.l2norm(curl)**2.
    b = st.sint(np.sum(-vt.lap(v) * v, 0))
    assert np.allclose(a, b)


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
    print(l2norm)
    print(h1norm)
    print(hm1norm)
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


def test_integral_of_sinkx():
    L = 10.5
    N = 128
    k = 2.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[0])
    st = ScalarTool(N, L)
    assert np.isclose(st.sint(th), 0.0)


def test_integral_of_one():
    L = 10.5
    N = 128
    integrand = np.ones((N, N))
    st = ScalarTool(N, L)
    assert np.isclose(st.sint(integrand), L**2.0)


def test_ifft_of_fft_equals_original_scalar_function():
    L = 10.5
    N = 128
    k = 2.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[0])
    st = ScalarTool(N, L)
    assert np.allclose(st.ifft(st.fft(th)), th)


def test_ifft_of_fft_equals_original_vector_function():
    L = 10.5
    N = 128
    k = 2.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    u = np.zeros((2, N, N))
    u[0] = np.sin(k * X[1])
    u[1] = np.cos(k * X[0])

    vt = VectorTool(N, L)
    assert np.allclose(vt.ifft(vt.fft(u)), u)


def test_that_l2norm_of_vector_with_siny_for_xcomponent_equals_sqrt_of_half_of_Lsq():
    L = 2.0 * np.pi
    N = 128
    vt = VectorTool(N, L)
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    u = np.zeros((2, N, N))
    u[0] = np.sin(2.0 * np.pi * X[1] / L)
    assert np.isclose(vt.l2norm(u), (0.5 * L ** 2)**0.5)


def test_that_is_incompressible_function_returns_true_for_cellular_flow():
    N = 128
    L = 2 * np.pi
    vt = VectorTool(N, L)
    u = np.zeros((2, N, N))
    u[0] = np.sin((2.0 * np.pi / L) * vt.X[0]) * \
        np.sin((2.0 * np.pi / L) * vt.X[1])
    u[1] = np.cos((2.0 * np.pi / L) * vt.X[0]) * \
        np.cos((2.0 * np.pi / L) * vt.X[1])
    assert vt.is_incompressible(u)


def test_that_div_free_projector_converts_a_compressible_flow_to_incompressible():
    N = 128
    L = 2 * np.pi
    vt = VectorTool(N, L)
    u = np.zeros((2, N, N))
    # Compressible flow
    u[0] = np.sin((2.0 * np.pi / L) * vt.X[0]) * \
        np.sin((2.0 * np.pi / L) * vt.X[1])
    u[1] = -np.cos((2.0 * np.pi / L) * vt.X[0]) * \
        np.cos((2.0 * np.pi / L) * vt.X[1])

    w = np.zeros((2, N, N))

    w[0] = np.sin((2.0 * np.pi / L) * vt.X[0]) * \
        np.sin((2.0 * np.pi / L) * vt.X[1])
    w[1] = np.cos((2.0 * np.pi / L) * vt.X[0]) * \
        np.cos((2.0 * np.pi / L) * vt.X[1])

    u = u + 0.1 * w

    assert vt.is_incompressible(vt.div_free_proj(u))


def test_validate_div_free_projector():
    L = 2.0 * np.pi
    N = 128
    st = ScalarTool(N, L)
    vt = VectorTool(N, L)
    u = np.zeros((2, N, N))
    # Compressible flow
    u[0] = np.sin((2.0 * np.pi / L) * vt.X[0]) * \
        np.sin((2.0 * np.pi / L) * vt.X[1])
    u[1] = -np.cos((2.0 * np.pi / L) * vt.X[0]) * \
        np.cos((2.0 * np.pi / L) * vt.X[1])

    w = np.zeros((2, N, N))

    w[0] = np.sin((2.0 * np.pi / L) * vt.X[0]) * \
        np.sin((2.0 * np.pi / L) * vt.X[1])
    w[1] = np.cos((2.0 * np.pi / L) * vt.X[0]) * \
        np.cos((2.0 * np.pi / L) * vt.X[1])

    u = u + 0.1 * w

    projection_alt = u - st.grad_invlap(vt.div(u))
    print(np.amax(abs(projection_alt - vt.div_free_proj(u))))
    assert np.allclose(projection_alt, vt.div_free_proj(u))


def test_lap_invlap_of_u_is_u_minus_mean_value():
    L = 2.0 * np.pi
    N = 128
    vt = VectorTool(N, L)
    u = np.zeros((2, N, N))
    # Compressible flow
    u[0] = np.sin((2.0 * np.pi / L) * vt.X[0]) * \
        np.sin((2.0 * np.pi / L) * vt.X[1]) + 2.0
    u[1] = -np.cos((2.0 * np.pi / L) * vt.X[0]) * \
        np.cos((2.0 * np.pi / L) * vt.X[1]) + 1.0

    assert np.allclose(vt.lap(vt.invlap(u)), vt.subtract_mean(u))


def test_subtract_mean_of_sinkx_plus_constant_equals_sinkx():
    L = 2.0
    N = 128
    st = ScalarTool(N, L)
    sinkx = np.sin((2 * np.pi / L) * st.X[0])
    th = sinkx + 10.0
    assert np.allclose(st.subtract_mean(th), sinkx)


def test_subtract_mean_of_velocity():
    L = 2.0
    N = 128
    vt = VectorTool(N, L)
    u = np.zeros((2, N, N))
    sinkx = np.sin((2 * np.pi / L) * vt.X[0])
    u[0] = sinkx + 2.0
    u[1] = sinkx + 1.0

    assert np.allclose(vt.subtract_mean(u), np.array([sinkx, sinkx]))


def test_st_ifft_of_function_is_real():
    L = 10.5
    N = 128
    k = 2.0 * np.pi / L
    X = np.mgrid[:N, :N].astype(float) * (L / N)
    th = np.sin(k * X[0])
    st = ScalarTool(N, L)
    assert np.all(np.isreal(st.ifft(st.fft(th))))


def test_get_spectrum_of_sinky():
    L = 10.5
    N = 128
    k = int(N / 4) * 2.0 * np.pi / L
    st = ScalarTool(N, L)
    th = np.sin(k * st.X[1])  # SIN(K Y)
    [klist, spectrum] = st.get_spectrum(th)
    spectrum_expected = np.zeros(len(klist))
    spectrum_expected[int(N / 4)] = 1.0
    assert np.allclose(spectrum, spectrum_expected)


def test_get_spectrum_of_sinkx():
    L = 10.5
    N = 128
    k = int(N / 4) * 2.0 * np.pi / L
    st = ScalarTool(N, L)
    th = np.sin(k * st.X[0])  # SIN(K X)
    [klist, spectrum] = st.get_spectrum(th)
    spectrum_expected = np.zeros(len(klist))
    spectrum_expected[int(N / 4)] = 1.0
    assert np.allclose(spectrum, spectrum_expected)


def test_get_spectrum_includes_Nyquist_mode():
    L = 10.5
    N = 128
    k = int(N / 2) * 2.0 * np.pi / L
    st = ScalarTool(N, L)
    th = np.cos(k * st.X[0])  # SIN(K X)
    [klist, spectrum] = st.get_spectrum(th)
    spectrum_expected = np.zeros(len(klist))
    spectrum_expected[int(N / 2)] = 2.0
    assert np.allclose(spectrum, spectrum_expected)


def test_check_no_spectral_blocking_negative_codition():
    N = 64
    L = 1.0
    st = ScalarTool(N, L)
    th = np.zeros((N, N))
    assert (not st.isblocked(th))


def test_check_no_spectral_blocking_positive_codition():
    N = 64
    L = 1.0
    st = ScalarTool(N, L)
    th = np.sin((N / 2 - 2) * (2. * np.pi / L) * st.X[0])
    assert st.isblocked(th)
