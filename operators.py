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
        self.u_sin_flow_hat = self.vt.fft(self.u_sin_flow)
        self.u_uniform_flow = np.zeros((2, self.N, self.N))
        self.u_uniform_flow[0] = np.ones((self.N, self.N))

    def adv_diff_op(self, u, th):
        op = self.st.dealias(-np.sum(u * self.st.grad(th), 0) +
                             self.kappa * self.st.lap(th))
        op = np.real(op)
        return op

    def adv_diff_op_hat(self, u_hat, th_hat):
        u = self.vt.ifft(u_hat * self.vt.dealias_array)
        th = self.st.ifft(th_hat * self.st.dealias_array)
        op_hat = - self.st.fft(np.sum(u * self.st.grad(th), 0)) - \
            self.kappa * self.st.K2 * (2 * np.pi / self.L)**2.0 * th_hat
        return op_hat * self.st.dealias_array

    def sin_flow_op(self, th):
        return self.adv_diff_op(self.u_sin_flow, th)

    def sin_flow_op_hat(self, th_hat):
        return self.adv_diff_op_hat(self.u_sin_flow_hat, th_hat)

    def uniform_flow_op(self, th):
        return self.adv_diff_op(self.u_uniform_flow, th)

    def lit_energy_op(self, th, U):
        return self.adv_diff_op(self.u_lit_energy(th, U), th)

    def u_lit_energy(self, th, U):
        v = th * self.st.grad_invlap(th)
        v = self.vt.div_free_proj(v)
        return U * self.L * v / self.vt.l2norm(v)

    def lit_energy_op_hat(self, th_hat, U):
        th = self.st.ifft(th_hat)

        grad_invlap_th = self.vt.ifft(-1.0j * self.st.KoverK2 *
                                      (2 * np.pi / self.L)**(-1.0) * th_hat)

        v = th * grad_invlap_th

        v = self.vt.div_free_proj(v)

        v = U * self.L * v / self.vt.l2norm(v)

        grad_th = self.vt.ifft(
            1.0j * self.st.K * (2 * np.pi / self.L) * th_hat)

        op_hat = self.st.fft(np.sum(-v * grad_th, 0)) - self.kappa * \
            self.st.K2 * (2 * np.pi / self.L)**2.0 * th_hat

        return op_hat

    def u_lit_enstrophy(self, th, gamma):
        v = th * self.st.grad_invlap(th)
        v = -self.vt.invlap(self.vt.div_free_proj(v))
        return gamma * self.L * v / self.st.l2norm(self.vt.curl(v))


class InputError(Exception):
    """ Error base case"""

    def __init__(self, message):
        self.message = message
