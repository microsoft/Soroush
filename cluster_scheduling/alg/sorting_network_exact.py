from collections import defaultdict

import numpy as np
import cvxpy as cp

from datetime import datetime
from gavel.scheduler.job_id_pair import JobIdPair
from scripts.problem import Problem


def get_rates(problem: Problem,
              epsilon: float = 0.99,
              break_down: bool = False):
    """ Compute the exact max-min fair solution using a single optimization.

    Args:
        problem: a description of a cluster scheduling problem  (e.g., available GPUs).
        epsilon: assign rewards to users based on this.
        break_down: if True, returns the break down of the run-time.

    Returns:
        job_id_to_job_rate_mapping: a mapping from job id to its assignment from each GPU.
        dur: time to find the allocation.
    """
    st_time = datetime.now()

    job_details = problem.sparse_job_list
    gpu_cap = problem.capacity_vector

    num_jobs = len(job_details)
    num_gpus = len(gpu_cap)

    throughput_coeff = np.empty((num_jobs, num_gpus))
    scale_factor_vector = np.empty((num_jobs, num_gpus))
    for jid, (priority_weight, scale_factor, throughput_list) in job_details:
        throughput_coeff[jid] = throughput_list * scale_factor / (priority_weight * np.average(throughput_list))
        scale_factor_vector[jid, :] = scale_factor

    obj_coeff_vector = np.power(epsilon, np.arange(num_jobs))

    allocation_vars = cp.Variable((num_jobs, num_gpus))
    effective_throughput = cp.sum(cp.multiply(throughput_coeff, allocation_vars), axis=1)

    if break_down:
        checkpoint_1 = datetime.now()

    output = _get_pairwise_sorting_network_constraints(num_jobs)
    final_vars = output[0]
    min_const_list_lh = output[1]
    min_const_list_rh = output[2]
    bin_const_s1 = output[5]
    bin_const_s2 = output[6]
    bin_const_s3 = output[7]
    bin_const_s4 = output[8]
    max_alloc_var_id = output[9]

    opt_vars = cp.hstack([effective_throughput, cp.Variable(max_alloc_var_id - num_jobs)])
    sorted_vars = opt_vars[final_vars]

    objective = cp.Maximize(cp.sum(cp.multiply(obj_coeff_vector, sorted_vars)))
    constraints = [allocation_vars >= 0,
                   cp.sum(cp.multiply(scale_factor_vector, allocation_vars), axis=0) <= gpu_cap,
                   cp.sum(allocation_vars, axis=1) <= 1,
                   opt_vars[min_const_list_lh] <= opt_vars[min_const_list_rh],
                   opt_vars[bin_const_s1] + opt_vars[bin_const_s2] <= opt_vars[bin_const_s3] + opt_vars[bin_const_s4]]

    model = cp.Problem(objective, constraints)
    if break_down:
        checkpoint_2 = datetime.now()
    _ = model.solve(solver=cp.GUROBI, verbose=True, Method=2, QCPDual=0, Crossover=0)
    end_time = datetime.now()
    if break_down:
        dur = ((checkpoint_1 - st_time).total_seconds(), (checkpoint_2 - checkpoint_1).total_seconds(), (end_time - checkpoint_2).total_seconds())
    else:
        dur = (end_time - st_time).total_seconds()
    final_job_allocation_matrix = allocation_vars.value
    job_id_to_job_rate_mapping = defaultdict(dict)
    for jid, _ in job_details:
        for gid, gpu in enumerate(problem.gpu_list):
            job_id_to_job_rate_mapping[JobIdPair(jid, None)][gpu] = final_job_allocation_matrix[jid, gid]
    print(f"duration optimization + sorting network {num_jobs}: {dur}")
    return job_id_to_job_rate_mapping, dur


def _get_pairwise_sorting_network_constraints(num_jobs):
    final_vars = np.arange(num_jobs)
    curr_max_alloc_var_id = num_jobs
    min_const_list_lh = []
    min_const_list_rh = []
    max_const_list_lh = []
    max_const_list_rh = []
    bin_const_s1 = []
    bin_const_s2 = []
    bin_const_s3 = []
    bin_const_s4 = []

    a = 1
    while a < num_jobs:
        b = a
        c = 0
        while b < num_jobs:
            _create_comparison_constraints(a, b, min_const_list_lh, min_const_list_rh, max_const_list_lh,
                                           max_const_list_rh, bin_const_s1, bin_const_s2, bin_const_s3, bin_const_s4, final_vars,
                                           curr_max_alloc_var_id)
            curr_max_alloc_var_id += 2
            b += 1
            c += 1
            if c >= a:
                c = 0
                b += a
        a *= 2

    a = a // 4
    e = 1
    while a > 0:
        d = e
        while d > 0:
            b = (d + 1) * a
            c = 0
            while b < num_jobs:
                _create_comparison_constraints(d * a, b, min_const_list_lh, min_const_list_rh, max_const_list_lh,
                                               max_const_list_rh, bin_const_s1, bin_const_s2, bin_const_s3,
                                               bin_const_s4, final_vars, curr_max_alloc_var_id)
                curr_max_alloc_var_id += 2
                c += 1
                b += 1
                if c >= a:
                    c = 0
                    b += a
            d = d // 2
        a = a // 2
        e = 2 * e + 1

    output = (final_vars, min_const_list_lh, min_const_list_rh,
              max_const_list_lh, max_const_list_rh, bin_const_s1,
              bin_const_s2, bin_const_s3, bin_const_s4, curr_max_alloc_var_id)
    return output


def _create_comparison_constraints(a, b, min_const_list_lh, min_const_list_rh, max_const_list_lh, max_const_list_rh,
                                   bin_const_s1, bin_const_s2, bin_const_s3, bin_const_s4, final_vars, curr_max_alloc_var_id):
    t1 = curr_max_alloc_var_id
    t2 = curr_max_alloc_var_id + 1
    sorted_args = np.sort([b, b - a])
    min_const_list_lh.append(t1)
    min_const_list_lh.append(t1)
    min_const_list_rh.append(final_vars[b - a])
    min_const_list_rh.append(final_vars[b])
    max_const_list_lh.append(final_vars[b])
    max_const_list_lh.append(final_vars[b - a])
    max_const_list_rh.append(t2)
    max_const_list_rh.append(t2)

    bin_const_s3.append(final_vars[b])
    bin_const_s4.append(final_vars[b - a])
    bin_const_s1.append(t1)
    bin_const_s2.append(t2)
    final_vars[sorted_args[0]] = t1
    final_vars[sorted_args[1]] = t2
