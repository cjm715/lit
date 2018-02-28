import tools
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os
import pickle
import pprint


class sol(object):
    def __init__(self):
        self.M = 0  # Number of timepts
        self.N = 0  # Number of spatial grid points in a single dimension
        self.T = 0.  # Final time
        self.dt = 0.
        self.Pe = 0.  # Peclet
        self.L = 0.  # Length of box side
        self.hist_th_hm1 = []
        self.hist_th_l2 = []
        self.hist_th_h1 = []
        self.hist_time = []
        self.hist_th = []
        self.hist_u = []
        self.hist_u_h1 = []
        self.hist_u_l2 = []


def sim(N=128, M=10000, T=13.0, L=1.0, gamma=1.0, Pe=1024, T_kick=0.01,
        save_every=100, pickle_file=None, plot=False):

    def f(th, u):
        return -1.0 * np.sum(vt.dealias(u) * st.grad(st.dealias(th)), 0) + kappa * st.lap(st.dealias(th))
        # return -1.0*np.sum(u*st.grad(th),0) + kappa * st.lap(th)

    def f_lit(th):
        return f(th, u_lit(th))

    def u_lit(th):
        u_lit = st.dealias(th) * vt.dealias(st.grad_invlap(th))
        u_lit = - vt.invlap(vt.div_free_proj(u_lit))
        u_lit = gamma * L * u_lit / st.l2norm(vt.curl(u_lit))
        return u_lit

    # Parameters
    h = L / N
    kappa = 1. / Pe
    print('Pe = ', Pe)
    if np.isinf(Pe):
        dt_cfl = (L / N) / (gamma * L)
        print('dt (CFL) = ', dt_cfl)
    else:
        length_min = 0.25 * 1. / (Pe)**0.5
        M_boyd = L / length_min
        N_boyd = int(2**np.ceil(np.log2(4 * (M_boyd - 1) + 6)))
        dt_cfl = min((L / N)**2. / kappa, (L / N) / (gamma * L))
        print('dt (CFL) = ', dt_cfl)
        print('N (Boyd) = ', N_boyd)
    print('N =', N)

    # ## Double precision
    ftype = np.float64
    ctype = np.complex128
    total_steps = M
    dt = T / M
    print('dt = ', dt)
    final_time_ind = total_steps
    total_time_pts = total_steps + 1
    time_array = np.linspace(0, T, total_time_pts, dtype=ftype)

    X = np.mgrid[:N, :N].astype(ftype) * h
    Nf = N // 2 + 1
    kx = np.fft.fftfreq(N, 1. / N).astype(int)
    ky = kx[:Nf].copy()
    ky[-1] *= -1
    K = np.array(np.meshgrid(kx, ky, indexing='ij'), dtype=int)

    st = tools.ScalarTool(N, L)
    vt = tools.VectorTool(N, L)

    temp_folder = 'temp/'
    os.system('mkdir ' + temp_folder)

    th = np.zeros((total_time_pts, N, N))
    th0 = np.sin(2. * np.pi * X[0] / L)
    th[0] = copy.copy(th0)

    # Initial kick
    # The default value for T_kick equal to 0.01 is sufficient to
    # initiate LIT optimization.

    num_steps_kick = max(round(T_kick / dt), 10)
    dt_kick = T_kick / num_steps_kick

    u_kick = np.zeros((2, N, N), dtype=ftype)
    u_kick[0, :, :] = np.sin(2. * np.pi * X[1] / L)

    for i in range(num_steps_kick):
        k1 = f(th[0], u_kick)
        k2 = f(th[0] + 0.5 * dt_kick * k1, u_kick)
        k3 = f(th[0] + 0.5 * dt_kick * k2, u_kick)
        k4 = f(th[0] + dt_kick * k3, u_kick)
        th[0] = th[0] + dt_kick * (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    time = 0.0
    hist_time = [time]

    hist_th = [th[0]]
    hist_th_hm1 = [st.hm1norm(th[0])]
    hist_th_l2 = [st.l2norm(th[0])]
    hist_th_h1 = [st.h1norm(th[0])]

    u = u_lit(th[0])
    hist_u = [u]
    hist_u_h1 = [vt.h1norm(u)]
    hist_u_l2 = [vt.l2norm(u)]
    if plot:
        plt.figure()
        st.plot(th[0])
        plt.title('time = %2.3f' % time_array[0])
        plt.show()

    u0 = copy.copy(u)
    assert total_steps == M
    for i in range(total_steps):
        k1 = f_lit(th[i])
        k2 = f_lit(th[i] + 0.5 * dt * k1)
        k3 = f_lit(th[i] + 0.5 * dt * k2)
        k4 = f_lit(th[i] + dt * k3)
        th[i + 1] = th[i] + dt * (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += dt

        if np.mod(i, save_every) == 0:
            hist_time.append(time)

            hist_th.append(th[i + 1])
            hist_th_hm1.append(st.hm1norm(th[i + 1]))
            hist_th_l2.append(st.l2norm(th[i + 1]))
            hist_th_h1.append(st.h1norm(th[i + 1]))

            u = u_lit(th[i + 1])
            hist_u.append(u)
            hist_u_h1.append(vt.h1norm(u))
            hist_u_l2.append(vt.l2norm(u))
            if plot:
                plt.figure()
                st.plot(th[i + 1])
                plt.title('time = %2.3f' % time_array[i + 1])
                plt.show()

                vt.plot(u)
                plt.show()

    sol_save = sol()
    sol_save.M = total_time_pts
    sol_save.N = N
    sol_save.T = T
    sol_save.dt = dt
    sol_save.Pe = Pe
    sol_save.L = L

    sol_save.hist_time = hist_time

    sol_save.hist_th = hist_th
    sol_save.hist_th_hm1 = hist_th_hm1
    sol_save.hist_th_l2 = hist_th_l2
    sol_save.hist_th_h1 = hist_th_h1

    sol_save.hist_u = hist_u
    sol_save.hist_u_h1 = hist_u_h1
    sol_save.hist_u_l2 = hist_u_l2

    if pickle_file != None:
        output = open(pickle_file, 'wb')
        pickle.dump(sol_save, output)

    return sol_save


def movie(time, scalar_hist, N, L, output_path='output/'):
    os.system('mkdir ' + output_path)
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


if __name__ == "__main__":

    N = int(sys.argv[0])
    Pe = int(sys.argv[1])
    pickle_file = "pe=" + str(Pe) + ".pkl"

    sim(N=N, M=1000, T=13.0, L=1.0, gamma=1.0, Pe=1024,
        save_every=10, pickle_file=pickle_file, plot=False)
