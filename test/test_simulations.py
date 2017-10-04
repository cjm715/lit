from simulations import lit_enstrophy_sim
from sol_checking import sol_lit_enstrophy_checker
from tools import dt_cfl
import numpy as np


def test_lit_enstrophy_sim():
    Pe = 128.0
    L = 1.0
    gamma = 1.0
    kappa = 1. / Pe
    T = 0.0001

    N_largest = 128  # largest number of modes

    dt_stable_for_N_largest = dt_cfl(N_largest, L, kappa, gamma * L)
    M_smallest = int(np.ceil(T / dt_stable_for_N_largest))
    # M_smallest = 1
    M_largest = 16 * M_smallest  # largest number of time steps

    sol_N_collection = []
    for N in [N_largest // 16, N_largest // 8, N_largest // 4, N_largest // 2]:
        # print(N)
        sol_N_collection.append(lit_enstrophy_sim(
            N, L, Pe, T, M_largest, cfl=False))

    sol_M_collection = []
    for M in [M_largest // 16, M_largest // 8, M_largest // 4, M_largest // 2]:
        # print(M)
        sol_M_collection.append(lit_enstrophy_sim(
            N_largest, L, Pe, T, M, cfl=False))

    best_solution = lit_enstrophy_sim(
        N_largest, L, Pe, T, M_largest, cfl=False)
    sol_N_collection.append(best_solution)
    sol_M_collection.append(best_solution)

    assert sol_lit_enstrophy_checker(
        sol_N_collection, sol_M_collection, Pe=Pe, L=L, gamma=gamma, T=T)
