import numpy as np
import matplotlib.pyplot as plt


class ScalarTool(object):
    """
    Description:
    ScalarTool contains a collection of functions necessary to compute basic
    operations such as gradients and norms on scalars defined on a 2D periodic
    square domain of length L and discretized in each dimension into N
    intervals.

    Inputs:
    N - number of discretized points in each dimension
    L - length of side
    """

    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.h = self.L / self.N
        self.X = np.mgrid[:self.N, :self.N].astype(float) * self.h
        self.kx = np.fft.fftfreq(self.N, 1. / self.N).astype(float)
        self.ky = np.fft.fftfreq(self.N, 1. / self.N).astype(float)
        self.K = np.array(np.meshgrid(
            self.kx, self.ky, indexing='ij'), dtype=int)
        self.K2 = sum(self.K * self.K, 0).astype(float)
        self.KoverK2 = self.K.astype(
            float) / np.where(self.K2 == 0, 1, self.K2).astype(float)

    def l2norm(self, scalar):
        self.scalar_size_test(scalar)

        return np.sum(np.ravel(scalar)**2.0 * self.h**2.0)**0.5

    def grad(self, scalar):
        self.scalar_size_test(scalar)

        scalar_hat = np.fft.fftn(scalar)
        return np.real(np.fft.ifftn(1.0j * self.K * (2 * np.pi / self.L) * scalar_hat, axes=(1, 2)))

    def h1norm(self, scalar):
        self.scalar_size_test(scalar)
        grad_scalar = self.grad(scalar)
        grad_scalar_sq = sum(grad_scalar * grad_scalar, 0)
        integrand = np.ravel(grad_scalar_sq)
        return np.sum(integrand * self.h**2.0)**0.5

    def lap(self, scalar):
        self.scalar_size_test(scalar)
        scalar_hat = np.fft.fftn(scalar)
        return np.real(np.fft.ifftn((-1.0) * self.K2 * (2 * np.pi / self.L)**2.0 * scalar_hat))

    def grad_invlap(self, scalar):
        self.scalar_size_test(scalar)
        scalar_hat = np.fft.fftn(scalar)
        return np.real(np.fft.ifftn(-1.0j * self.KoverK2 * (2 * np.pi / self.L)**(-1.0) * scalar_hat, axes=(1, 2)))

    def hm1norm(self, scalar):
        self.scalar_size_test(scalar)
        grad_invlap_scalar = self.grad_invlap(scalar)
        grad_invlap_scalar_sq = sum(grad_invlap_scalar *
                                    grad_invlap_scalar, 0)  # dot product
        integrand = np.ravel(grad_invlap_scalar_sq)
        return np.sum(integrand * self.h**2.0)**0.5

    def plot_scalar(self, scalar):
        self.scalar_size_test(scalar)
        im = plt.imshow(np.transpose(scalar),
                        cmap=plt.cm.gray,
                        extent=(0, self.L, 0, self.L))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(im)
        plt.show()

    def scalar_size_test(self, scalar):
        if np.shape(scalar) != (self.N, self.N):
            raise InputError("Scalar field array does not have correct shape")


class InputError(Exception):
    """ Error base case"""

    def __init__(self, message):
        self.message = message
