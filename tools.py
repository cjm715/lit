import numpy as np
import matplotlib.pyplot as plt


def create_grid(N, L):
    return np.mgrid[:N, :N].astype(float) * (L / N)


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
        self.dealias_array = (abs(self.kx[:, np.newaxis]) < self.N / 3.0) * \
            (abs(self.ky[np.newaxis, :]) < self.N / 3.0)

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

    def plot(self, scalar):
        self.scalar_size_test(scalar)
        im = plt.imshow(np.transpose(scalar),
                        cmap=plt.cm.gray,
                        extent=(0, self.L, 0, self.L))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(im)

    def scalar_size_test(self, scalar):
        if np.shape(scalar) != (self.N, self.N):
            raise InputError("Scalar field array does not have correct shape")

    def sint(self, scalar):
        """ Performs spatial integration """
        self.scalar_size_test(scalar)
        return np.sum(np.ravel(scalar) * self.h**2.0)

    def dealias(self, scalar):
        """ Perform 1/3 dealias on scalar """
        self.scalar_size_test(scalar)
        temp_hat = self.fft(scalar) * self.dealias_array
        return self.ifft(temp_hat)

    def fft(self, scalar):
        """ Performs fft of scalar field """
        self.scalar_size_test(scalar)
        return np.fft.fftn(scalar)

    def ifft(self, scalar_hat):
        """ Performs inverse fft of scalar field """
        self.scalar_size_test(scalar_hat)
        return np.fft.ifftn(scalar_hat)


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
        self.kx = np.fft.fftfreq(self.N, 1. / self.N).astype(float)
        self.ky = np.fft.fftfreq(self.N, 1. / self.N).astype(float)
        self.K = np.array(np.meshgrid(
            self.kx, self.ky, indexing='ij'), dtype=int)
        self.K2 = sum(self.K * self.K, 0).astype(float)
        self.KoverK2 = self.K.astype(
            float) / np.where(self.K2 == 0, 1, self.K2).astype(float)
        self.dealias_array = (abs(self.kx[:, np.newaxis]) < self.N / 3.0) * \
            (abs(self.ky[np.newaxis, :]) < self.N / 3.0)

    def fft(self, vector):
        """ Performs fft of vector field """
        self.vector_size_test(vector)
        return np.fft.fftn(vector, axes=(1, 2))

    def ifft(self, vector_hat):
        """ Performs inverse fft of vector hat field """
        self.vector_size_test(vector_hat)
        return np.fft.ifftn(vector_hat, axes=(1, 2))

    def plot(self, vector, high_quality=False):
        """ Plots a quiver plot of the vector field """
        self.vector_size_test(vector)
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
                       vector[1][1:-1:m, 1:-1:m])
        plt.quiverkey(
            Q, 0.8, 1.03, 2, r'%.2f $\frac{m}{s}$' % np.amax(vector), labelpos='E',)
        plt.title('Vector field')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.xlim(0.0, self.L)
        plt.ylim(0.0, self.L)
        plt.axis('scaled')

    def vector_size_test(self, vector):
        """ Determines if vector is correct size """
        if np.shape(vector) != (2, self.N, self.N):
            raise InputError("Vector field array does not have correct shape")


class InputError(Exception):
    """ Input Error """

    def __init__(self, message):
        self.message = message
