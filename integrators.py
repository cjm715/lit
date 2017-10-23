import numpy as np
import pyfftw


def FE_timestepper(op, th0, dt):
    th = th0 + dt * op(th0)
    return th


def RK4_timestepper(op, th0, dt):
    k1 = op(th0)
    k2 = op(th0 + 0.5 * dt * k1)
    k3 = op(th0 + 0.5 * dt * k2)
    k4 = op(th0 + dt * k3)
    th = th0 + dt * (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return th


def FE(op, th0, tarray):
    return integrator(op, FE_timestepper, th0, tarray)


def RK4(op, th0, tarray):
    return integrator(op, RK4_timestepper, th0, tarray)


def integrator(op, timestepper, th0, tarray):
    num_time_steps = len(tarray)
    shape = np.shape(th0)
    N = shape[0]
    if np.iscomplexobj(th0):
        th = pyfftw.empty_aligned(
            (num_time_steps, N, N // 2 + 1), dtype=complex)
    else:
        th = pyfftw.empty_aligned(
            (num_time_steps, N, N), dtype=float)
    for i, t in enumerate(tarray):
        if i == 0:
            th[0] = th0
        else:
            dt = tarray[i] - tarray[i - 1]
            th[i] = timestepper(op, th[i - 1], dt)
        # print('Time: ', tarray[i])
    return th


def integrator2(op, timestepper, th0, tarray, dt0_cfl):
    num_time_steps = len(tarray)
    shape = np.shape(th0)
    N = shape[0]
    num_time_steps = len(tarray)
    if np.iscomplexobj(th0):
        th = pyfftw.empty_aligned(
            (num_time_steps, N, N // 2 + 1), dtype=complex)
    else:
        th = pyfftw.empty_aligned(
            (num_time_steps, N, N), dtype=float)
    for i, t in enumerate(tarray):
        if i == 0:
            th[0] = th0
        else:
            dt = tarray[i] - tarray[i - 1]
            M = np.ceil(dt / dt0_cfl).astype('int') if dt > dt0_cfl else 1
            # print(M)
            # pdb.set_trace()
            th[i] = timestepper(op, th[i - 1], dt, M)

        #print('Time: ', tarray[i])
    return th


def mega_RK4_timestepper(op, th0, dt, M):
    th = np.copy(th0)
    for i in range(M):
        th = RK4_timestepper(op, th, dt / M)
    return th
