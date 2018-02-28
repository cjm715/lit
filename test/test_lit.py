import tools
import lit
import numpy as np
import math


def test_fourth_order_convergence_of_entrophy_constained_lit_simulation():
    s = []
    ss = []
    N = 256
    L = 1.0
    M_list = [1, 2, 4]
    T = 0.01
    Pe = 512

    dt_list = []
    for M in M_list:
        s.append(lit.sim(N=N, M=M, T=T, L=L, save_every=1,
                         T_kick=0.01, Pe=Pe, plot=False))
        dt_list.append(T / M)
    error_list = []
    st = tools.ScalarTool(N, L)
    for i in range(len(M_list) - 1):
        error_list.append(st.l2norm(s[i].hist_th[-1] - s[2].hist_th[-1]))

    error_array = np.array(error_list)
    dt_array = np.array(dt_list)
    #print(st.l2norm(s[-3].hist_th[-1] - s[-2].hist_th[-1]))
    #print(st.l2norm(s[-2].hist_th[-1] - s[-1].hist_th[-1]))
    R = st.l2norm(s[0].hist_th[-1] - s[1].hist_th[-1]) / \
        st.l2norm(s[1].hist_th[-1] - s[2].hist_th[-1])
    p = np.log(R) / np.log(2)
    print('p = ', p)
    assert abs(p - 4) < 0.5
