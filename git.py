import tools
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os
import pickle
import pprint
import lit


def increase_res(scalar, N_grid):
    st_grid = tools.ScalarTool(N_grid, L)
    th_hat = np.fft.rfft(scalar, axis=1)

    th_hat1 = np.zeros((N, N_grid // 2 + 1), dtype=ctype)
    th_hat1[:, 0:(N // 2 + 1)] = N_grid / N * th_hat

    th_hat2 = np.fft.fft(th_hat1, axis=0)

    th_hat3 = np.zeros((N_grid, N_grid // 2 + 1), dtype=ctype)
    th_hat3[0:N // 2, :] = N_grid / N * th_hat2[0:N // 2, :]
    th_hat3[(N_grid - N // 2):, :] = N_grid / N * th_hat2[N // 2:, :]
    th_new = st_grid.ifft(th_hat3)
    return th_new


def f(th, u):
    th_d = st.dealias(th)
    return st.dealias(-1.0 * np.sum(vt.dealias(u) * st.grad(th_d), 0)
                      + kappa * st.lap(th_d))


def g(phi, u):
    phi_d = st.dealias(phi)
    return st.dealias(-1.0 * np.sum(vt.dealias(u) * st.grad(phi_d), 0)
                      - kappa * st.lap(phi_d))


def line_search(u, d, eta_array):
    J_array = np.zeros(np.shape(eta_array))
    for i, eta in enumerate(eta_array):
        J_array[i] = get_obj(normalize(div_free_proj(dealias(u + eta * d))))
    plt.figure()
    plt.loglog(eta_array, J_array)
    plt.show()
    return eta_array[np.argmin(J_array)]


def get_obj(u):
    # Forward integration
    th = integrate_forward(u, th0)
    return 0.5 * st.hm1norm(th[M - 1])**2


def integrate_forward(u, th0):
    th = np.zeros((M, N, N))
    th[0] = th0
    for i in range(M - 1):
        # Heun's method
        k1 = f(th[i], u[i])
        th_euler = th[i] + dt * k1
        th[i + 1] = th[i] + 0.5 * dt * (k1 + f(th_euler, u[i + 1]))
    return th


def integrate_backward(u, phiT):
    phi = np.zeros((M, N, N))
    phi[M - 1] = phiT
    for i in reversed(range(M - 1)):
        # Heun's method
        k1 = g(phi[i + 1], u[i + 1])
        phi_euler = phi[i + 1] - dt * k1
        phi[i] = phi[i + 1] - 0.5 * dt * (k1 + g(phi_euler, u[i]))
    return phi


def compute_gradJ_and_J(u):
    # Forward integration
    th = integrate_forward(u, th0)

    # Compute objective
    obj = 0.5 * st.hm1norm(th[M - 1])**2

    # Backward integration
    phiT = st.invlap(th[M - 1])
    phi = integrate_backward(u, phiT)

    # Compute gradient
    grad = np.zeros((M, 2, N, N))
    for i in range(M):
        grad[i] = st.dealias(phi[i]) * vt.dealias(st.grad(th[i]))
        grad[i] = vt.div_free_proj(vt.dealias(grad[i]))
    lapu = lap(u)
    mu = dot(lapu, grad) / dot(lapu, lapu)
    grad = grad - mu * lapu

    return grad, obj


def compute_d(u):
    # Forward integration
    th = integrate_forward(u, th0)

    # Compute objective
    obj = 0.5 * st.hm1norm(th[M - 1])**2

    # Backward integration
    phiT = st.invlap(th[M - 1])
    phi = integrate_backward(u, phiT)

    # Compute d
    d = np.zeros((M, 2, N, N))
    for i in range(M):
        d[i] = st.dealias(phi[i]) * vt.dealias(st.grad(th[i]))
        d[i] = vt.invlap(vt.div_free_proj(vt.dealias(d[i])))
    d = normalize(d) - u

    return d


def lap(v):
    lapv = np.zeros((M, 2, N, N))
    for i in range(M):
        lapv[i] = vt.lap(v[i])
    return lapv


def invlap(v):
    invlapv = np.zeros((M, 2, N, N))
    for i in range(M):
        invlapv[i] = vt.invlap(v[i])
    return invlapv


def normalize(v):
    return v * (gamma * L) / mean_enstrophy(v)**0.5


def mean_enstrophy(v):
    integ = 0
    for i in range(M):
        integ += vt.h1norm(v[i])**2. * dt
    integ = (1. / T) * integ
    return integ


def is_incompressible(v):
    cond = True
    for i in range(M):
        cond = cond * vt.is_incompressible(v[i])
    return cond == 1


def div_free_proj(v):
    for i in range(M):
        v[i] = vt.div_free_proj(vt.dealias(v[i]))
    return v


def dealias(v):
    for i in range(M):
        v[i] = vt.dealias(v[i])
    return v


def dot(v, u):
    dot = 0
    for i in range(M):
        dot += st.sint(sum(v[i] * u[i], 0)) * dt

    return dot


if __name__ == "__main__":
    N = 64
    M = 1000
    L = 1.0
    h = L / N
    T = 3.0
    dt = T / M
    kappa = 0.0
    gamma = 1.0
    ftype = np.float64
    ctype = np.complex128

    st = tools.ScalarTool(N, L)
    vt = tools.VectorTool(N, L)

    sol_lit = lit.sim(N=N, M=M - 1, Pe=np.inf, plot=False,
                      T=T, save_th_every=M - 1, save_u_every=1)
    u = np.array(sol_lit.hist_u)
    th0 = sol_lit.hist_th[0]
    print(mean_enstrophy(u))
    time_array = sol_lit.hist_u_time

    eta_array = np.array([0.1, .01, 0.001])
    num_iterations = 100
    for i in range(num_iterations):
        gradJ, J = compute_gradJ_and_J(u)
        d = compute_d(u)
        eta = line_search(u, d, eta_array)
        u = normalize(div_free_proj(dealias(u + eta * d)))
        print('eta=', eta,
              'mag of gradJ = ', dot(gradJ, gradJ)**0.5,
              'mag d = ', dot(d, d)**0.5,
              'J=', J,
              'incompressible?', is_incompressible(u),
              'mean enstrophy', mean_enstrophy(u))
