from collections import defaultdict

import numpy as np
import cvxpy as cp

from datetime import datetime
from gavel.scheduler.job_id_pair import JobIdPair
from scripts.problem import Problem


def get_pairwise_sorting_network_constraints(num_jobs):
    final_vars = np.arange(num_jobs)
    # for var in effective_throughput:
    #     final_vars.append(var)
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
            create_comparison_constraints(a, b, min_const_list_lh, min_const_list_rh, max_const_list_lh,
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
                create_comparison_constraints(d * a, b, min_const_list_lh, min_const_list_rh, max_const_list_lh,
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
    return final_vars, min_const_list_lh, min_const_list_rh, max_const_list_lh, max_const_list_rh, \
           bin_const_s1, bin_const_s2, bin_const_s3, bin_const_s4, curr_max_alloc_var_id


# def get_bitonic_sorting_network(num_jobs):
#     final_vars = np.arange(num_jobs)
#     # for var in effective_throughput:
#     #     final_vars.append(var)
#     curr_max_alloc_var_id = num_jobs
#     min_const_list_lh = []
#     min_const_list_rh = []
#     max_const_list_lh = []
#     max_const_list_rh = []
#     bin_const_s1 = []
#     bin_const_s2 = []
#     bin_const_s3 = []
#     bin_const_s4 = []
#
#     k = 2
#     while k <= num_jobs:
#         j = k // 2
#         while j > 0:
#             for i in range(num_jobs):
#                 l = np.bitwise_xor(i, j)
#                 create_comparison_constraints(i, l, min_const_list_lh, min_const_list_rh, max_const_list_lh,
#                                               max_const_list_rh, bin_const_s1, bin_const_s2, bin_const_s3,
#                                               bin_const_s4, final_vars, curr_max_alloc_var_id)
#                 curr_max_alloc_var_id += 2
#             j = j // 2
#         k *= 2
#
#     return final_vars, min_const_list_lh, min_const_list_rh, max_const_list_lh, max_const_list_rh, \
#            bin_const_s1, bin_const_s2, bin_const_s3, bin_const_s4, curr_max_alloc_var_id


def create_comparison_constraints(a, b, min_const_list_lh, min_const_list_rh, max_const_list_lh, max_const_list_rh,
                                  bin_const_s1, bin_const_s2, bin_const_s3, bin_const_s4, final_vars, curr_max_alloc_var_id):
    # t1 = cp.Variable(1)
    # t2 = cp.Variable(1)
    t1 = curr_max_alloc_var_id
    t2 = curr_max_alloc_var_id + 1
    sorted_args = np.sort([b, b - a])
    # const_list.append(t1 <= final_vars[b - a])
    # const_list.append(t1 <= final_vars[b])
    min_const_list_lh.append(t1)
    min_const_list_lh.append(t1)
    min_const_list_rh.append(final_vars[b - a])
    min_const_list_rh.append(final_vars[b])
    # d1 = cp.Variable(1, boolean=True)
    # d2 = cp.Variable(1, boolean=True)
    # d1 = curr_max_bin_var_id
    # d2 = curr_max_bin_var_id + 1
    # const_list.append(t2 <= final_vars[b] + (1 - d1) * M)
    # const_list.append(t2 <= final_vars[b - a] + (1 - d2) * M)
    # const_list.append(d1 + d2 == 1)
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


def get_rates(problem: Problem, break_down=False):
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

    obj_coeff_vector = np.power(0.99, np.arange(num_jobs))

    allocation_vars = cp.Variable((num_jobs, num_gpus))
    effective_throughput = cp.sum(cp.multiply(throughput_coeff, allocation_vars), axis=1)

    if break_down:
        checkpoint_1 = datetime.now()

    output = get_pairwise_sorting_network_constraints(num_jobs)
    final_vars, min_const_list_lh, min_const_list_rh, max_const_list_lh, max_const_list_rh, \
    bin_const_s1, bin_const_s2, bin_const_s3, bin_const_s4, max_alloc_var_id = output

    opt_vars = cp.hstack([effective_throughput, cp.Variable(max_alloc_var_id - num_jobs)])
    # bin_vars = cp.Variable(max_bin_var_id, boolean=True)
    sorted_vars = opt_vars[final_vars]

    objective = cp.Maximize(cp.sum(cp.multiply(obj_coeff_vector, sorted_vars)))
    constraints = [allocation_vars >= 0,
                   cp.sum(cp.multiply(scale_factor_vector, allocation_vars), axis=0) <= gpu_cap,
                   cp.sum(allocation_vars, axis=1) <= 1,
                   opt_vars[min_const_list_lh] <= opt_vars[min_const_list_rh],
                   # opt_vars[max_const_list_lh] <= opt_vars[max_const_list_rh],
                   opt_vars[bin_const_s1] + opt_vars[bin_const_s2] <= opt_vars[bin_const_s3] + opt_vars[bin_const_s4]]
                   # opt_vars[max_const_list_lh] <= opt_vars[max_const_list_rh] + (1 - bin_vars[max_const_list_bin]) * M,
                   # bin_vars[bin_const_s1] + bin_vars[bin_const_s2] == 1]
    # constraints.extend(sorted_constraints)

    model = cp.Problem(objective, constraints)
    if break_down:
        checkpoint_2 = datetime.now()
    result = model.solve(solver=cp.GUROBI, verbose=True, Method=2, QCPDual=0, Crossover=0)
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
