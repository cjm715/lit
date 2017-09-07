import numpy as np
from tools import ScalarTool, VectorTool
import pyfftw


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

        # Temporary arrays
        self.th = pyfftw.empty_aligned((self.N, self.N), dtype=float)
        self.grad_invlap_th = pyfftw.empty_aligned(
            (2, self.N, self.N), dtype=float)
        self.v = pyfftw.empty_aligned((2, self.N, self.N), dtype=float)
        self.grad_th = pyfftw.empty_aligned((2, self.N, self.N), dtype=float)
        self.out_hat = pyfftw.empty_aligned((self.N, self.N), dtype=complex)

    def adv_diff_op(self, u, th):
        op = -np.sum(u * self.st.grad(th), 0) + self.kappa * self.st.lap(th)
        op = np.real(op)
        return op

    def adv_diff_op_hat(self, u_hat, th_hat):
        u = self.vt.ifft(u_hat)
        th = self.st.ifft(th_hat)
        op_hat = - self.st.fft(np.sum(u * self.st.grad(th), 0)) - \
            self.kappa * self.st.K2 * (2 * np.pi / self.L)**2.0 * th_hat
        return op_hat

    def sin_flow_op(self, th):
        return self.adv_diff_op(self.u_sin_flow, th)

    def sin_flow_op_hat(self, th_hat):
        return self.adv_diff_op_hat(self.u_sin_flow_hat, th_hat)

    def uniform_flow_op(self, th):
        return self.adv_diff_op(self.u_uniform_flow, th)

    def lit_energy_op(self, th, U):

        th_hat = self.st.fft(th)

        grad_invlap_th = self.vt.ifft(-1.0j * self.st.KoverK2 *
                                      (2 * np.pi / self.L)**(-1.0) * th_hat)
        v = th * grad_invlap_th
        v = self.vt.div_free_proj(v)
        v = U * self.L * v / self.vt.l2norm(v)

        grad_th = self.vt.ifft(
            1.0j * self.st.K * (2 * np.pi / self.L) * th_hat)

        lap_th = self.st.ifft((-1.0) * self.st.K2 *
                              (2 * np.pi / self.L)**2.0 * th_hat)

        op = -np.sum(v * grad_th, 0) + self.kappa * lap_th

        return op

    def u_lit_energy(self, th, U):
        v = th * self.st.grad_invlap(th)
        v = self.vt.div_free_proj(v)
        return U * self.L * v / self.vt.l2norm(v)

    # @profile
    # @jit(numba.complex128[:, :](numba.complex128[:, :], numba.float64))
    def lit_energy_op_hat(self, th_hat, U):

        self.th = self.st.ifft(th_hat)
        self.v = self.th * self.vt.ifft(
            -1.0j * self.st.KoverK2 * (2 * np.pi / self.L)**(-1.0) * th_hat)

        self.v -= self.vt.ifft(self.vt.KoverK2 *
                               np.sum(self.vt.K * self.vt.fft(self.v), 0))

        self.v *= U * self.L / self.vt.l2norm(self.v)

        self.grad_th = self.vt.ifft(
            1.0j * self.st.K * (2 * np.pi / self.L) * th_hat)

        self.out_hat = self.st.fft(np.sum(-self.v * self.grad_th, 0))
        self.out_hat -= self.kappa * self.st.K2 * \
            (2 * np.pi / self.L)**2.0 * th_hat

        return self.out_hat

    def u_lit_enstrophy(self, th, gamma):
        v = th * self.st.grad_invlap(th)
        v = -self.vt.invlap(self.vt.div_free_proj(v))
        return gamma * self.L * v / self.st.l2norm(self.vt.curl(v))

    def lit_enstrophy_op(self, th, gamma, dealias=False):
        th_hat = self.st.fft(th)

        grad_invlap_th = self.vt.ifft(-1.0j * self.st.KoverK2 *
                                      (2 * np.pi / self.L)**(-1.0) * th_hat)
        v = th * grad_invlap_th
        v = -self.vt.invlap(self.vt.div_free_proj(v))
        v = gamma * self.L * v / self.st.l2norm(self.vt.curl(v))

        grad_th = self.vt.ifft(
            1.0j * self.st.K * (2 * np.pi / self.L) * th_hat)

        lap_th = self.st.ifft((-1.0) * self.st.K2 *
                              (2 * np.pi / self.L)**2.0 * th_hat)

        op = np.sum(-v * grad_th, 0) + self.kappa * lap_th

        if dealias is True:
            op = self.st.dealias(op)

        return op

    def lit_enstrophy_op_hat(self, th_hat, gamma):
        th = self.st.ifft(th_hat)
        grad_invlap_th = self.vt.ifft(-1.0j * self.st.KoverK2 *
                                      (2 * np.pi / self.L)**(-1.0) * th_hat)

        v = th * grad_invlap_th
        v = -self.vt.invlap(self.vt.div_free_proj(v))
        v = gamma * self.L * v / self.st.l2norm(self.vt.curl(v))

        grad_th = self.vt.ifft(
            1.0j * self.st.K * (2 * np.pi / self.L) * th_hat)

        op_hat = self.st.fft(np.sum(-v * grad_th, 0)) - self.kappa * \
            self.st.K2 * (2 * np.pi / self.L)**2.0 * th_hat

        return op_hat


class InputError(Exception):
    """ Error base case"""

    def __init__(self, message):
        self.message = message
