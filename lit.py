import tools
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os
import sys
import pickle
import pprint


class sol(object):
    def __init__(self):
        self.M = 0  # Number of total steps
        self.N = 0  # Number of spatial grid points in a single dimension
        self.T = 0.  # Final time
        self.dt = 0.
        self.Pe = 0.  # Peclet
        self.L = 0.  # Length of box side
        self.hist_th_hm1 = []
        self.hist_th_l2 = []
        self.hist_th_h1 = []
        self.hist_th_time = []
        self.hist_th = []

        self.hist_u = []
        self.hist_u_time = []
        self.hist_u_h1 = []
        self.hist_u_l2 = []


def sim(N=128, M=1000, T=1.0, L=1.0, gamma=1.0, U=1.0, Pe=1024,
        T_kick=0.01, save_th_every=10, save_u_every=10, pickle_file=None,
        plot=False, constraint='enstrophy'):

    def f(th, u):
        th_d = st.dealias(th)
        return st.dealias(-1.0 * np.sum(vt.dealias(u) * st.grad(th_d), 0)
                          + kappa * st.lap(th_d))

    def f_lit(th):
        return f(th, u_lit(th))

    def u_lit_enstrophy(th):
        th_d = st.dealias(th)
        u_lit = th_d * st.grad_invlap(th_d)
        u_lit = - vt.invlap(vt.div_free_proj(u_lit))
        u_lit = gamma * L * u_lit / st.l2norm(vt.curl(u_lit))
        return u_lit

    def u_lit_energy(th):
        u_lit = st.dealias(th) * vt.dealias(st.grad_invlap(th))
        u_lit = vt.div_free_proj(u_lit)
        u_lit = U * L * u_lit / vt.l2norm(u_lit)
        return u_lit

    if constraint == 'enstrophy':
        u_lit = u_lit_enstrophy
    elif constraint == 'energy':
        u_lit = u_lit_energy

        # Parameters
    h = L / N
    kappa = 1. / Pe

    # ## Double precision
    ftype = np.float64
    ctype = np.complex128
    total_steps = M
    dt = T / M
    print('dt = ', dt)
    final_time_ind = total_steps
    total_time_pts = total_steps + 1

    X = np.mgrid[:N, :N].astype(ftype) * h
    Nf = N // 2 + 1
    kx = np.fft.fftfreq(N, 1. / N).astype(int)
    ky = kx[:Nf].copy()
    ky[-1] *= -1
    K = np.array(np.meshgrid(kx, ky, indexing='ij'), dtype=int)

    st = tools.ScalarTool(N, L)
    vt = tools.VectorTool(N, L)

    th0 = np.sin(2. * np.pi * X[0] / L)
    th = copy.copy(th0)

    # Initial kick
    # The default value for T_kick equal to 0.01 is sufficient to
    # initiate LIT optimization.

    num_steps_kick = int(max(round(T_kick / dt), 10))
    dt_kick = T_kick / num_steps_kick

    u_kick = np.zeros((2, N, N), dtype=ftype)
    u_kick[0, :, :] = np.sin(2. * np.pi * X[1] / L)

    for i in range(num_steps_kick):
        k1 = f(th, u_kick)
        k2 = f(th + 0.5 * dt_kick * k1, u_kick)
        k3 = f(th + 0.5 * dt_kick * k2, u_kick)
        k4 = f(th + dt_kick * k3, u_kick)
        th = th + dt_kick * (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    time = 0.0

    hist_th = [th]
    hist_th_time = [time]
    hist_th_hm1 = [st.hm1norm(th)]
    hist_th_l2 = [st.l2norm(th)]
    hist_th_h1 = [st.h1norm(th)]

    u = u_lit(th)
    hist_u = [u]
    hist_u_time = [time]
    hist_u_h1 = [vt.h1norm(u)]
    hist_u_l2 = [vt.l2norm(u)]
    if plot:
        plt.figure()
        st.plot(th)
        plt.title('time = %2.3f' % time)
        plt.show()

    u0 = copy.copy(u)
    assert total_steps == M

    for i in range(1, total_steps + 1):
        k1 = f_lit(th)
        k2 = f_lit(th + 0.5 * dt * k1)
        k3 = f_lit(th + 0.5 * dt * k2)
        k4 = f_lit(th + dt * k3)
        th = th + dt * (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += dt

        if np.mod(i, save_th_every) == 0:

            hist_th.append(th)
            hist_th_time.append(time)
            hist_th_hm1.append(st.hm1norm(th))
            hist_th_l2.append(st.l2norm(th))
            hist_th_h1.append(st.h1norm(th))

            if plot:
                plt.figure()
                st.plot(th)
                plt.title('time = %2.3f' % time)
                plt.show()

                vt.plot(u)
                plt.show()

        if np.mod(i, save_u_every) == 0:
            u = u_lit(th)
            hist_u.append(u)
            hist_u_time.append(time)
            hist_u_h1.append(vt.h1norm(u))
            hist_u_l2.append(vt.l2norm(u))

    sol_save = sol()
    sol_save.M = total_time_pts
    sol_save.N = N
    sol_save.T = T
    sol_save.dt = dt
    sol_save.Pe = Pe
    sol_save.L = L

    sol_save.hist_th = hist_th
    sol_save.hist_th_time = hist_th_time
    sol_save.hist_th_hm1 = hist_th_hm1
    sol_save.hist_th_l2 = hist_th_l2
    sol_save.hist_th_h1 = hist_th_h1

    sol_save.hist_u = hist_u
    sol_save.hist_u_time = hist_u_time
    sol_save.hist_u_h1 = hist_u_h1
    sol_save.hist_u_l2 = hist_u_l2

    if pickle_file != None:
        output = open(pickle_file, 'wb')
        pickle.dump(sol_save, output)

    return sol_save


def movie(time, scalar_hist, N, L, output_path='output/'):
    os.system('mkdir ' + output_path)
    os.system('mkdir ' + output_path + 'images/')
    st = tools.ScalarTool(N, L)
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
              "image%04d.png -c:v libx264 -pix_fmt yuv420p " + output_path +
              "movies.mp4")

    # os.system('rm -r ' + output_path + 'images/')


def compute_norms(scalar_hist, N, L):
    st = tools.ScalarTool(N, L)
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

    # PLEASE READ BEFORE RUNNING ON FLUX SERVER!
    #
    # Transfer this script lit.py , tools.py, and flux_pbs_enstrophy.sh (or flux_pbs_energy.sh)
    # to flux.
    #
    # This is specifically built to interface with the pbs scrpts to work on
    # univeristy of Michigan's flux server. Make sure to use the following command
    # `qsub -t 1-27 flux_pbs_enstrophy.sh' where 27 is chosen since it is the (Length
    # of Pe_list[=9]) times (number of trials per Peclet[=3]). Similarly
    # for the energy constraint run `qsub -t 1-33 flux_pbs_energy.sh'

    sim_num = int(sys.argv[1])
    constraint = str(sys.argv[2])
    T = float(sys.argv[3])

    print(sim_num)

    if constraint == "enstrophy":
        Pe_list = [128.0, 256.0, 512.0, 1024.0,
                   2048.0, 4096.0, 8192.0, 16384.0, np.inf]
    elif constraint == "energy":
        Pe_list = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
                   64.0, 128.0, 256.0, 512.0, np.inf]

    Pe = Pe_list[(sim_num - 1) // 3]
    kappa = 1.0 / Pe

    # THESE PARAMETERS SHOULD NOT BE CHANGE!
    L = 1.0
    U = 1.0
    gamma = 1.0

    # Determine N and dt_cfl
    if constraint == "enstrophy":
        if Pe == np.inf:
            N = 512
            dt_cfl = 0.25 * (L / N) / (gamma * L)
        else:
            lb = (kappa / gamma)**0.5
            l_smallest = 0.25 * lb  # a quarter of batchelor scale
            print('lb = ', lb)
            num_wavelengths = L / l_smallest
            print('N_boyd = ', tools.N_boyd(num_wavelengths))
            N = min(tools.N_boyd(num_wavelengths), 512)
            dt_cfl = 0.25 * min((L / N)**2. / kappa, (L / N) / (gamma * L))

    elif constraint == "energy":

        if Pe == np.inf:
            N = 512
            dt_cfl = 0.25 * (L / N) / (U)
        else:
            lb = (kappa / U)
            l_smallest = 0.25 * lb  # a quarter of batchelor scale
            print('lb = ', lb)
            num_wavelengths = L / l_smallest
            print('N_boyd = ', tools.N_boyd(num_wavelengths))
            N = min(tools.N_boyd(num_wavelengths), 512)
            dt_cfl = 0.25 * min((L / N)**2. / kappa, (L / N) / U)

    print('N = ', N)
    print('dt CFL = ', dt_cfl)

    # Determine M_list given dt_cfl
    # Run 3 different simulations with 2 and 4 times as many time num_steps
    # This will be used to calculate convergence metrics.
    M0 = round(T / dt_cfl)  # approx number of time steps according to CFL
    M0 = int(2**np.ceil(np.log2(M0)))  # make power of two
    M_list = [M0, int(2 * M0), int(4 * M0)]

    # Select M based off of sim_num
    M_index = (sim_num - 1) % 3
    M = M_list[M_index]

    output_folder = "output-pe=" + str(Pe) + "-M=" + str(M) + "/"
    os.system('mkdir ' + output_folder)
    pickle_file = output_folder + "pe=" + str(Pe) + "-M=" + str(M) + ".pkl"

    save_th_every = max(int(round(M / 128)), 1)
    save_u_every = max(int(round(M / 8)), 1)

    solution = sim(N=N, M=M, T=T, L=L, U=U, gamma=gamma, Pe=Pe, constraint=constraint,
                   save_th_every=save_th_every, save_u_every=save_u_every,
                   pickle_file=pickle_file, plot=False)

    movie(solution.hist_th_time, solution.hist_th,
          N, L, output_path=output_folder)

    plt.figure()

    st = tools.ScalarTool(N, L)
    st.plot(solution.hist_th[-1])
    plt.savefig(output_folder + 'plot_final_frame-pe=' + str(Pe) + '.png')

    plt.figure()
    plot_norms(solution.hist_th_time, solution.hist_th, N, L)
    plt.savefig(output_folder + 'plot_norms-pe=' + str(Pe) + '.png')
