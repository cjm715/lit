import numpy as np
import matplotlib.pyplot as plt
from tools import ScalarTool, VectorTool
import pyfftw


def compute_norms(scalar_hist, N, L):
    st = ScalarTool(N, L)
    time_length, _, _ = np.shape(scalar_hist)

    hm1norm_hist = np.zeros(time_length)
    l2norm_hist = np.zeros(time_length)
    h1norm_hist = np.zeros(time_length)

    for i, scalar in enumerate(scalar_hist):
        hm1norm_hist[i] = st.hm1norm(scalar)
        l2norm_hist[i] = st.l2norm(scalar)
        h1norm_hist[i] = st.h1norm(scalar)

    return [hm1norm_hist, l2norm_hist, h1norm_hist]


def plot_norms(time, scalar_hist, N, L, high_quality=False):
    if high_quality:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=12)
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif', size=12)
    hm1norm_hist, l2norm_hist, h1norm_hist = compute_norms(scalar_hist, N, L)
    plt.semilogy(time, hm1norm_hist,
                 label=r'$H^{-1}$', linestyle='-', color='k')
    plt.semilogy(time, l2norm_hist, label=r'$L^2$', linestyle='--', color='k')
    plt.semilogy(time, h1norm_hist, label=r'$H^{1}$', linestyle=':', color='k')
    plt.legend()
    plt.xlabel('Time')
    plt.grid(alpha=0.5)
