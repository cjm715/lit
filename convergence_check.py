import tools
import numpy as np
import copy
import time
import os
import pickle
import pprint
from lit import *

M_list = [4096, 8192, 16384]
Pe = 512.0
s = []
for M in M_list:
    pickle_file = "output-pe=" + str(Pe) + "-M=" + str(M) + \
    "/pe=" + str(Pe) + "-M=" + str(M) + ".pkl"

    with open(pickle_file, 'rb') as f:
        s.append(pickle.load(f, encoding='latin1'))


N = s[0].N
L = s[0].L
dt_list = [item.T / item.M for item in s]
error_list = []
st = tools.ScalarTool(N, L)
for i in range(len(M_list) - 1):
    error_list.append(st.l2norm(s[i].hist_th[-1] - s[-1].hist_th[-1]))

error_array = np.array(error_list)
dt_array = np.array(dt_list)
print(st.l2norm(s[-3].hist_th[-1] - s[-2].hist_th[-1]))
print(st.l2norm(s[-2].hist_th[-1] - s[-1].hist_th[-1]))
R = st.l2norm(s[-3].hist_th[-1] - s[-2].hist_th[-1]) / \
    st.l2norm(s[-2].hist_th[-1] - s[-1].hist_th[-1])
p = np.log(R) / np.log(2)
print('p = ', p)
