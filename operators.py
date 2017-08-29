import numpy as np
from tools import ScalarTool, VectorTool


class OperatorKit(object):
    def __init__(self, N, L, kappa):
        self.N = N
        self.L = L
        self.kappa = kappa
        self.st = ScalarTool(N, L)
        self.vt = VectorTool(N, L)
        self.u_sin_flow = np.zeros((2, self.N, self.N))
        self.u_sin_flow[0] = np.sin((2.0 * np.pi / self.L) * self.st.X[1])
        self.u_uniform_flow = np.zeros((2, self.N, self.N))
        self.u_uniform_flow[0] = np.ones((self.N, self.N))

    def adv_diff_op(self, u, th):
        op = - self.st.dealias(sum(u * self.st.grad(th), 0)
                               ) + self.kappa * self.st.lap(th)
        op = np.real(op)
        return op

    def adv_diff_op_hat(self, u_hat, th_hat):
        u = self.vt.ifft(u_hat)
        th = self.st.ifft(th_hat)
        op_hat = - self.st.ifft(sum(u * self.st.grad(th), 0)) - \
            self.kappa * self.st.K2 * (2 * np.pi / self.L)**2.0 * th_hat
        return op_hat * self.st.dealias_array

    def sin_flow_op(self, th):
        return self.adv_diff_op(self.u_sin_flow, th)

    def uniform_flow_op(self, th):
        return self.adv_diff_op(self.u_uniform_flow, th)


class InputError(Exception):
    """ Error base case"""

    def __init__(self, message):
        self.message = message
