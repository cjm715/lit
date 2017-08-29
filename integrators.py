import numpy as np


def FE_timestepper(op, th0, dt):
    th = th0 + dt * op(th0)
    return th


def FE(op, th0, tarray):
    num_time_steps = len(tarray)
    shape = np.shape(th0)
    N = shape[0]
    th = np.zeros((num_time_steps, N, N))
    for i, t in enumerate(tarray):
        if i == 0:
            th[0] = th0
        else:
            dt = tarray[i] - tarray[i - 1]
            th[i] = FE_timestepper(op, th[i - 1], dt)
    return th
