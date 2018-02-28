import tools
import lit
import numpy as np
import math


def test_fourth_order_convergence_of_entrophy_constained_lit_simulation():
    s = []
    N = 256
    L = 1.0
    M_list = [1, 2, 4]
    T = 0.01
    Pe = 512

    for M in M_list:
        s.append(lit.sim(N=N, M=M, T=T, L=L, save_every=1,
                         T_kick=0.01, Pe=Pe, plot=False))
    st = tools.ScalarTool(N, L)
    R = st.l2norm(s[0].hist_th[-1] - s[1].hist_th[-1]) / \
        st.l2norm(s[1].hist_th[-1] - s[2].hist_th[-1])
    p = np.log(R) / np.log(2)
    print('p = ', p)
    assert abs(p - 4) < 0.5


def test_fourth_order_convergence_of_energy_constained_lit_simulation():
    s = []
    N = 256
    L = 1.0
    M_list = [1, 2, 4]
    T = 0.001
    Pe = 512
    U = 1.0

    for M in M_list:
        s.append(lit.sim(N=N, M=M, U=U, T=T, L=L, save_every=1,
                         T_kick=0.01, Pe=Pe, plot=False, constraint="energy"))
    st = tools.ScalarTool(N, L)
    R = st.l2norm(s[0].hist_th[-1] - s[1].hist_th[-1]) / \
        st.l2norm(s[1].hist_th[-1] - s[2].hist_th[-1])
    p = np.log(R) / np.log(2)
    print('p = ', p)
    assert abs(p - 4) < 0.5
