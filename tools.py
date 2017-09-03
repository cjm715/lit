import numpy as np
import matplotlib.pyplot as plt
import pyfftw.interfaces.numpy_fft as fft
# from numpy import fft
from pyfftw.interfaces import cache

cache.enable()


def create_grid(N, L):
    return np.mgrid[:N, :N].astype(float) * (L / N)


def dt_cfl(N, L, kappa, U):
    return min(L / (N * U), L**2 / (N**2 * kappa))


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

        self.Nf = self.N // 2 + 1
        self.kx = np.fft.fftfreq(self.N, 1. / self.N).astype(int)
        self.ky = self.kx[:self.Nf].copy()
        self.ky[-1] *= -1
        self.K = np.array(np.meshgrid(
            self.kx, self.ky, indexing='ij'), dtype=int)
        self.K2 = np.sum(self.K * self.K, 0, dtype=int)
        self.KoverK2 = self.K.astype(
            float) / np.where(self.K2 == 0, 1, self.K2).astype(float)
        self.oneoverK2 = 1.0 / \
            np.where(self.K2 == 0.0, 1.0, self.K2).astype(float)
        self.mean_zero_array = self.K2 != 0.0
        self.kmax_dealias = 2. / 3. * (self.N / 2 + 1)
        self.dealias_array = np.array((abs(self.K[0]) < self.kmax_dealias) * (
            abs(self.K[1]) < self.kmax_dealias), dtype=bool)
        self.num_threads = 4

    def l2norm(self, scalar):
        self.scalar_input_test(scalar)
        return np.sum(np.ravel(scalar)**2.0 * self.h**2.0)**0.5

    def grad(self, scalar):
        self.scalar_input_test(scalar)

        scalar_hat = self.fft(scalar)
        return fft.irfftn(1.0j * self.K * (2 * np.pi / self.L) * scalar_hat, axes=(1, 2), threads=self.num_threads)

    def h1norm(self, scalar):
        self.scalar_input_test(scalar)
        grad_scalar = self.grad(scalar)
        grad_scalar_sq = np.sum(grad_scalar * grad_scalar, 0)
        integrand = np.ravel(grad_scalar_sq)
        return np.sum(integrand * self.h**2.0)**0.5

    def lap(self, scalar):
        self.scalar_input_test(scalar)
        scalar_hat = self.fft(scalar)
        return self.ifft((-1.0) * self.K2 * (2 * np.pi / self.L)**2.0 * scalar_hat)

    def grad_invlap(self, scalar):
        self.scalar_input_test(scalar)
        scalar_hat = self.fft(scalar)
        return fft.irfftn(-1.0j * self.KoverK2 * (2 * np.pi / self.L)**(-1.0) * scalar_hat, axes=(1, 2), threads=self.num_threads)

    def hm1norm(self, scalar):
        self.scalar_input_test(scalar)
        grad_invlap_scalar = self.grad_invlap(scalar)
        grad_invlap_scalar_sq = np.sum(grad_invlap_scalar *
                                       grad_invlap_scalar, 0)  # dot product
        integrand = np.ravel(grad_invlap_scalar_sq)
        return np.sum(integrand * self.h**2.0)**0.5

    def plot(self, scalar):
        self.scalar_input_test(scalar)
        im = plt.imshow(np.transpose(scalar),
                        cmap=plt.cm.gray,
                        extent=(0, self.L, 0, self.L),
                        origin="lower")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(im)

    def scalar_input_test(self, scalar):
        if np.shape(scalar) != (self.N, self.N):
            print(np.shape(scalar))
            raise InputError("Scalar field array does not have correct shape.")
        if not np.all(np.isrealobj(scalar)):
            raise InputError("Scalar field array should be real.")

    def scalar_hat_input_test(self, scalar_hat):
        if np.shape(scalar_hat) != (self.N, self.Nf):
            print(np.shape(scalar_hat))
            raise InputError("Scalar field array does not have correct shape.")

    def sint(self, scalar):
        """ Performs spatial integration """
        self.scalar_input_test(scalar)
        return np.sum(np.ravel(scalar) * self.h**2.0)

    def dealias(self, scalar):
        """ Perform 1/3 dealias on scalar """
        self.scalar_input_test(scalar)
        temp_hat = self.fft(scalar) * self.dealias_array
        return self.ifft(temp_hat)

    def fft(self, scalar):
        """ Performs fft of scalar field """
        self.scalar_input_test(scalar)
        scalar_hat = fft.rfftn(scalar, threads=self.num_threads)
        return scalar_hat

    def ifft(self, scalar_hat):
        """ Performs inverse fft of scalar field """
        self.scalar_hat_input_test(scalar_hat)
        scalar = fft.irfftn(scalar_hat, threads=self.num_threads)
        return scalar

    def subtract_mean(self, scalar):
        """ subtract off mean """
        self.scalar_input_test(scalar)
        scalar_hat = self.fft(scalar)
        return np.real(self.ifft(scalar_hat * self.mean_zero_array))


class VectorTool(object):
    """
    Description:
    VectorTool contains a collection of functions necessary to compute basic
    operations such as norms on scalars defined on a 2D periodic
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
        self.Nf = self.N // 2 + 1
        self.kx = np.fft.fftfreq(self.N, 1. / self.N).astype(int)
        self.ky = self.kx[:self.Nf].copy()
        self.ky[-1] *= -1
        self.K = np.array(np.meshgrid(
            self.kx, self.ky, indexing='ij'), dtype=int)
        self.K2 = np.sum(self.K * self.K, 0, dtype=int)
        self.KoverK2 = self.K.astype(
            float) / np.where(self.K2 == 0, 1, self.K2).astype(float)
        self.oneoverK2 = 1.0 / \
            np.where(self.K2 == 0.0, 1.0, self.K2).astype(float)
        self.mean_zero_array = self.K2 != 0.0
        self.kmax_dealias = 2. / 3. * (self.N / 2 + 1)
        self.dealias_array = np.array((abs(self.K[0]) < self.kmax_dealias) * (
            abs(self.K[1]) < self.kmax_dealias), dtype=bool)
        self.num_threads = 1

    def div(self, vector):
        """ Take divergence of vector """
        self.vector_input_test(vector)
        vector_hat = self.fft(vector)
        return fft.irfftn(np.sum(1j * self.K * (2 * np.pi / self.L) * vector_hat, 0), threads=self.num_threads)

    def fft(self, vector):
        """ Performs fft of vector field """
        self.vector_input_test(vector)
        vector_hat = fft.rfftn(vector, axes=(1, 2), threads=self.num_threads)
        return vector_hat

    def ifft(self, vector_hat):
        """ Performs inverse fft of vector hat field """
        self.vector_hat_input_test(vector_hat)
        vector = fft.irfftn(vector_hat, axes=(1, 2), threads=self.num_threads)

        return vector

    def plot(self, vector, high_quality=False):
        """ Plots a quiver plot of the vector field """
        self.vector_input_test(vector)
        if high_quality:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=12)
        else:
            plt.rc('text', usetex=False)
            plt.rc('font', family='sans-serif', size=12)

        m = max(round(self.N / 25), 1)
        Q = plt.quiver(self.X[0][1:-1:m, 1:-1:m],
                       self.X[1][1:-1:m, 1:-1:m],
                       vector[0][1:-1:m, 1:-1:m],
                       vector[1][1:-1:m, 1:-1:m], linewidths=2.0)
        plt.quiverkey(
            Q, 0.8, 1.03, 2, r'%.2f $\frac{m}{s}$' % np.amax(vector), labelpos='E',)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title('')
        plt.xlim(0.0, self.L)
        plt.ylim(0.0, self.L)
        plt.axis('scaled')

    def dealias(self, vector):
        """ Dealias vector """
        self.vector_input_test(vector)
        vector_hat = self.fft(vector)
        vector_hat = vector_hat * self.dealias_array
        return np.real(self.ifft(vector_hat))

    def l2norm(self, vector):
        """ L2 norm of a vector field """
        self.vector_input_test(vector)
        integrand = np.sum(vector * vector, 0)

        return np.sum(np.ravel(integrand) * self.h**2)**0.5

    def vector_input_test(self, vector):
        """ Determines if vector is correct size """
        if np.shape(vector) != (2, self.N, self.N):
            print(np.shape(vector))
            raise InputError("Vector field array does not have correct shape")

        if not np.all(np.isrealobj(vector)):
            raise InputError("Scalar field array should be real.")

    def vector_hat_input_test(self, vector_hat):
        """ Determines if vector is correct size """
        if np.shape(vector_hat) != (2, self.N, self.Nf):
            print(np.shape(vector_hat))
            raise InputError("Vector field array does not have correct shape")

    def is_incompressible(self, vector):
        self.vector_input_test(vector)
        return np.allclose(self.div(vector), 0)

    def div_free_proj(self, vector):
        """ performs leray divergence-free projection """
        self.vector_input_test(vector)
        vector_hat = self.fft(vector)
        return self.ifft(vector_hat - self.KoverK2 * np.sum(self.K * vector_hat, 0))

    def curl(self, vector):
        """ Perform curl of vector """
        self.vector_input_test(vector)
        vector_hat = self.fft(vector)
        w = fft.irfftn(
            1j * self.K[1] * vector_hat[0] - 1j * self.K[0] * vector_hat[1], threads=self.num_threads)
        return w

    def invlap(self, vector):
        """ Inverse laplacian of vector """
        self.vector_input_test(vector)
        vector_hat = self.fft(vector)
        return np.real(self.ifft(-1.0 * (2.0 * np.pi / self.L)**(-2.0) *
                                 self.oneoverK2 * self.mean_zero_array * vector_hat))

    def lap(self, vector):
        """ Laplacian of vector """
        self.vector_input_test(vector)
        vector_hat = self.fft(vector)
        return np.real(self.ifft(-1.0 * (2.0 * np.pi / self.L)**(2.0) * (self.K2) * vector_hat))

    def subtract_mean(self, vector):
        """ subtract off mean """
        self.vector_input_test(vector)
        vector_hat = self.fft(vector)
        return np.real(self.ifft(vector_hat * self.mean_zero_array))


class InputError(Exception):
    """ Input Error """

    def __init__(self, message):
        self.message = message
