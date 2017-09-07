from termcolor import colored
import numpy as np
from tools import ScalarTool
from post_processing import compute_norms
import matplotlib.pyplot as plt
from tools import ScalarTool


def sol_lit_enstrophy_checker(sol_N_collection, sol_M_collection, Pe, L, gamma, T):
    # last in collection is most resolved
    best_sol = sol_N_collection[-1]
    [_, best_sol_th_hist] = best_sol
    [M, N, _] = np.shape(best_sol_th_hist)
    [_, l2norm_best_sol, _] = compute_norms(best_sol_th_hist, N, L)

    # Check each criterion
    print('')
    print('')
    print(colored('====== Solution Checker Report ======', 'green'))
    print('Peclet = ', Pe)
    print('Box side length = ', L)
    print('Rate of strain = ', gamma)
    print('Final time = ', T)
    print('Number of time steps: ', M)
    print('Number of modes per side: ', N)
    print('')

    criteria = []

    [criterion, abs_error_N, rel_error_N] = check_N_convergence(
        sol_N_collection, L)
    print('N convergence check: ' + _criterion_report_msg(criterion))
    criteria.append(criterion)

    [criterion, abs_error_M, rel_error_M] = check_M_convergence(
        sol_M_collection, L)
    print('Time step convergence: ' + _criterion_report_msg(criterion))
    criteria.append(criterion)

    criterion = check_no_spectral_blocking(best_sol, L)
    print('No spectral blocking check: ' + _criterion_report_msg(criterion))
    criteria.append(criterion)

    criterion = check_solution_is_resolved_beyond_batchelor_scale(
        best_sol, Pe)
    print('Spatial resolution check: ' + _criterion_report_msg(criterion))
    criteria.append(criterion)

    criterion = check_solution_error_is_tolerable(
        abs_error_N, rel_error_N, abs_error_M, rel_error_M)
    print('Absolute and relative error tolerance check: ' +
          _criterion_report_msg(criterion))
    criteria.append(criterion)

    criterion = check_L2_decay_relation(best_sol, L, Pe)
    print('L2 decay relation: ' + _criterion_report_msg(criterion))
    criteria.append(criterion)

    # Report errors
    abs_error = abs_error_N + abs_error_M
    rel_error = (abs_error_N + abs_error_M) / l2norm_best_sol[-1]
    machine_eps = np.finfo(float).eps

    if abs_error == 0.0:
        abs_error = machine_eps
    if rel_error == 0.0:
        rel_error = machine_eps

    print('')
    print('Estimated absolute error of L2 Norm at final time: ', abs_error)
    print('Estimated relative error of L2 Norm at final time: ', rel_error)
    print('')
    print(colored('====== Solution Checker Report ======', 'green'))
    print('')

    return all(criteria)


def check_N_convergence(sol_N_collection, L):
    [_, sol_largest_N] = sol_N_collection[-1]
    [num_time_pts, N, _] = np.shape(sol_largest_N)
    [_, l2norm_hist_largest_N, _] = compute_norms(sol_largest_N, N, L)

    error_list = []
    N_list = []
    for [time, th_hist] in sol_N_collection[:-1]:
        [num_time_pts, N, _] = np.shape(th_hist)
        [_, l2norm_hist, _] = compute_norms(th_hist, N, L)
        error = abs(l2norm_hist[-1] - l2norm_hist_largest_N[-1])
        error_list.append(error)
        N_list.append(N)
        # print(N, error)
    check = _sequence_converges_to_zero(error_list)
    abs_error = abs(error_list[-1])
    rel_error = abs_error / l2norm_hist_largest_N[-1]
    return [check, abs_error, rel_error]


def check_M_convergence(sol_M_collection, L, tol=10**(-10)):
    [_, sol_largest_M] = sol_M_collection[-1]
    [num_time_pts, N, _] = np.shape(sol_largest_M)
    [_, l2norm_hist_largest_M, _] = compute_norms(sol_largest_M, N, L)

    error_list = []
    M_list = []
    for [time, th_hist] in sol_M_collection[:-1]:
        [M, N, _] = np.shape(th_hist)
        [_, l2norm_hist, _] = compute_norms(th_hist, N, L)
        error = abs(l2norm_hist[-1] - l2norm_hist_largest_M[-1])
        error_list.append(error)
        M_list.append(M)
        # print(M, error)
    check = _sequence_converges_to_zero(error_list, tol=tol)
    abs_error = abs(error_list[-1])
    rel_error = abs_error / l2norm_hist_largest_M[-1]

    return [check, abs_error, rel_error]


def check_no_spectral_blocking(sol, L):

    [_, th_hist] = sol
    [_, N, _] = np.shape(th_hist)
    st = ScalarTool(N, L)
    check = True
    for th in th_hist:
        check = check * (not st.isblocked(th))
    return check


def check_solution_is_resolved_beyond_batchelor_scale(sol, Pe, constraint='enstrophy'):
    [_, th_hist] = sol
    [_, N, _] = np.shape(th_hist)
    if constraint == 'enstrophy':
        return N > 4. * (np.sqrt(Pe) - 1) + 6  # John P. Boyd's rule-of-thumb
    elif constraint == 'energy':
        return N > 4. * (Pe - 1) + 6  # John P. Boyd's rule-of-thumb


def check_solution_error_is_tolerable(abs_error_N, rel_error_N, abs_error_M, rel_error_M, atol=10.**(-10), rtol=10**(-3)):
    return ((abs_error_N < atol and rel_error_N < rtol) and (abs_error_M < atol and rel_error_M < rtol))


def check_L2_decay_relation(sol, L, Pe):
    kappa = 1. / Pe
    [time, th_hist] = sol
    [_, N, _] = np.shape(th_hist)
    hm1norm_hist, l2norm_hist, h1norm_hist = compute_norms(th_hist, N, L)
    num_pts = len(time)
    l2decay = np.zeros(num_pts - 1)
    h1norm_dependence = -2.0 * kappa * h1norm_hist[:-1]**2

    for i in range(num_pts - 1):
        dt = time[i + 1] - time[i]
        l2decay[i] = (l2norm_hist[i + 1]**2 - l2norm_hist[i]**2) / dt

    return np.allclose(h1norm_dependence, l2decay, rtol=0.01)


def _criterion_report_msg(criterion):
    if criterion:
        msg = colored('PASSED', 'green')
    else:
        msg = colored('FAILED', 'red')
    return msg


def _sequence_converges_to_zero(error_list, tol=10.**(-10)):
    """ Checks that sequence a_n (error_list) converges to zero by
    checking that each pair a_n and a_n+1 decreases or each are both below
    the threshold tolerance.
    """
    check = True
    for i in range(len(error_list) - 1):
        error1_below_thresh = (error_list[i] < tol)
        error2_below_thresh = (error_list[i + 1] < tol)
        decreasing = (error_list[i + 1] < error_list[i])
        check_pair = decreasing or (
            error1_below_thresh and error2_below_thresh)
        check = check * check_pair
        # print(check_pair)
    # print(check)
    return check
