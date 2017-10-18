import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools import ScalarTool, VectorTool
import pyfftw

import os


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


def plot_norms(time, scalar_hist, N, L, high_quality=False, graph='log'):
    if high_quality:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=12)
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif', size=12)
    hm1norm_hist, l2norm_hist, h1norm_hist = compute_norms(scalar_hist, N, L)

    if graph == 'log':
        plt.semilogy(time, hm1norm_hist,
                     label=r'$H^{-1}$', linestyle='-', color='k')
        plt.semilogy(time, l2norm_hist, label=r'$L^2$',
                     linestyle='--', color='k')
        plt.semilogy(time, h1norm_hist,
                     label=r'$H^{1}$', linestyle=':', color='k')
    elif graph == 'linear':
        plt.plot(time, hm1norm_hist,
                 label=r'$H^{-1}$', linestyle='-', color='k')
        plt.plot(time, l2norm_hist, label=r'$L^2$',
                 linestyle='--', color='k')
        plt.plot(time, h1norm_hist,
                 label=r'$H^{1}$', linestyle=':', color='k')

    plt.legend()
    plt.xlabel('Time')
    plt.grid(alpha=0.5)


def movie(time, scalar_hist, N, L, output_path='output/'):
    os.system('mkdir ' + output_path + 'images/')
    st = ScalarTool(N, L)
    # st.plot(scalar_hist[i])
    # plt.savefig(outputPath + "image%.4d.png" % i, format='png')
    for i in range(len(time)):
        fig = plt.figure()
        st.plot(np.real(scalar_hist[i]))
        plt.title('Time = %.3f' % time[i])
        plt.savefig(output_path + 'images/' + "image%.4d.png" %
                    i, format='png')
        # plt.savefig("image.png", format='png')
        plt.close(fig)

    os.system("ffmpeg -y -framerate 20 -i " + output_path + 'images/'
              "image%04d.png -c:v libx264 -pix_fmt yuv420p " + output_path + "movies.mp4")

    # os.system('rm -r ' + output_path + 'images/')
