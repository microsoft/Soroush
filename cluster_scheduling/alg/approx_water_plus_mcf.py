from collections import defaultdict
from datetime import datetime

import numpy as np
import cvxpy as cp

from alg import approx_waterfilling
from alg import waterfilling_utils
from scripts.problem import Problem
from gavel.scheduler.job_id_pair import JobIdPair
from utilities import constants


def get_rates(problem: Problem, num_approx_iter, epsilon, k, alpha, beta, priority_aware=False, throughput_aware=False,
              break_down=False):
    st_time = datetime.now()

    output = waterfilling_utils.get_vectorized_characteristics(problem)
    normalized_throughput_coeff, throughput_coeff, scale_factor_vector, priority_vector = output
    job_allocation_matrix, dur = approx_waterfilling.get_rates(problem, num_approx_iter, priority_aware=priority_aware,
                                                               throughput_aware=throughput_aware, break_down=break_down,
                                                               return_matrix=True)

    job_details = problem.sparse_job_list
    output = compute_throughput_given_jobs(job_details, job_allocation_matrix, normalized_throughput_coeff,
                                           throughput_coeff, scale_factor_vector, problem.capacity_vector,
                                           epsilon=epsilon, k=k, alpha=alpha, beta=beta, break_down=break_down)
    dur = (datetime.now() - st_time).total_seconds()
    if break_down:
        final_job_allocation_matrix, detailed_dur = output
    else:
        final_job_allocation_matrix = output

    job_id_to_job_rate_mapping = defaultdict(dict)
    for jid, _ in job_details:
        for gid, gpu in enumerate(problem.gpu_list):
            job_id_to_job_rate_mapping[JobIdPair(jid, None)][gpu] = final_job_allocation_matrix[jid, gid]

    #### Checking the final solution
    throughput_coeff = np.empty_like(job_allocation_matrix)
    for jid, (priority_weight, scale_factor, throughput_list) in job_details:
        throughput_coeff[jid] = throughput_list
    final_thru = np.sum(np.multiply(throughput_coeff, final_job_allocation_matrix), axis=1)
    initial_thru = np.sum(np.multiply(throughput_coeff, job_allocation_matrix), axis=1)
    # assert np.all(final_thru >= initial_thru - constants.O_epsilon)
    if break_down:
        return job_id_to_job_rate_mapping, (dur, detailed_dur)
    return job_id_to_job_rate_mapping, dur


def compute_throughput_given_jobs(job_details, job_allocation_matrix, norm_throughput_coefficient,
                                  throughput_coefficient, scale_factor_vector, gpu_cap, epsilon, k, alpha, beta,
                                  num_jobs_per_barrier, break_down=False):
    if break_down:
        st_time_model = datetime.now()

    num_jobs = len(job_details)
    num_gpus = len(gpu_cap)

    final_throughput_coefficient = norm_throughput_coefficient.reshape(num_jobs, 1) * throughput_coefficient
    normalized_eff_thru = np.add.reduce(final_throughput_coefficient * job_allocation_matrix, axis=1)
    sorted_jids = np.argsort(normalized_eff_thru)
    num_bins = np.int(np.ceil(num_jobs / num_jobs_per_barrier))
    obj_coeff_vector = np.power(epsilon, np.arange(num_bins))
    sorted_job_to_bin = np.repeat(np.arange(num_bins, dtype=np.int32), num_jobs_per_barrier)[:num_jobs]
    multi_coeff_obj = np.empty(num_jobs)
    multi_coeff_obj[sorted_jids] = obj_coeff_vector[sorted_job_to_bin]
    additive_term = k * np.power(beta, np.arange(num_bins - 1, 0, -1))

    allocation_vars = cp.Variable((num_jobs, num_gpus))
    t_lb = cp.Variable(num_bins - 1)
    normalized_effective_throughput = cp.sum(cp.multiply(final_throughput_coefficient, allocation_vars), axis=1)
    total_effective_throughput = cp.sum(cp.multiply(throughput_coefficient, allocation_vars))
    objective = cp.Maximize((1 - alpha) * cp.sum(cp.multiply(multi_coeff_obj, normalized_effective_throughput)) +
                            alpha * total_effective_throughput)

    last_bin_starting_jidx = num_jobs_per_barrier * (num_bins - 1)
    ub_allowed_violation = additive_term[sorted_job_to_bin[:last_bin_starting_jidx]]
    ub_barrier_variables = t_lb[sorted_job_to_bin[:last_bin_starting_jidx]]
    lb_barrier_variables = t_lb[sorted_job_to_bin[num_jobs_per_barrier:] - 1]
    constraints = [
        allocation_vars >= 0,
        cp.sum(cp.multiply(scale_factor_vector, allocation_vars), axis=0) <= gpu_cap,
        cp.sum(allocation_vars, axis=1) <= 1,
        t_lb[:-1] <= t_lb[1:],
        lb_barrier_variables <= normalized_effective_throughput[sorted_jids[num_jobs_per_barrier:]],
        normalized_effective_throughput[sorted_jids[:last_bin_starting_jidx]] <= ub_barrier_variables + ub_allowed_violation,
        # normalized_effective_throughput[sorted_jids[:-1]] - normalized_effective_throughput[sorted_jids[1:]] <= additive_term
    ]
    # effective_throughput >= lb_rate]
    # for idx in range(num_jobs - 1):
    #     st_idx = sorted_jids[idx]
    #     en_idx = sorted_jids[idx + 1]
    #     constraints.append(effective_throughput[st_idx] <= effective_throughput[en_idx])

    model = cp.Problem(objective, constraints)
    if break_down:
        checkpoint1 = datetime.now()
    model.solve(
        solver=cp.GUROBI,
        Method=2,
        QCPDual=0,
        # Crossover=0
    )
    #Crossover=0, BarConvTol=1e-4
    if break_down:
        curr_time = datetime.now()
        time_model = (checkpoint1 - st_time_model).total_seconds()
        time_solve = (curr_time - checkpoint1).total_seconds()
        return allocation_vars.value, (time_model, time_solve)
    return allocation_vars.value


# def compute_throughput_given_jobs(job_details, job_allocation_matrix, gpu_cap, break_down=False):
#     if break_down:
#         st_time_model = datetime.now()
#
#     num_jobs = len(job_details)
#     num_gpus = len(gpu_cap)
#     allocation_vars = cp.Variable((num_jobs, num_gpus))
#
#     throughput_coeff = np.empty((num_jobs, num_gpus))
#     lb_rate = np.empty(num_jobs)
#     normalized_eff_thru = np.empty(num_jobs)
#     scale_factor_vector = np.empty((num_jobs, num_gpus))
#     for jid, (priority_weight, scale_factor, throughput_list) in job_details:
#         throughput_coeff[jid] = throughput_list / np.average(throughput_list)
#         lb_rate[jid] = np.sum(job_allocation_matrix[jid] * throughput_list)
#         normalized_eff_thru[jid] = lb_rate[jid] * scale_factor / (priority_weight * np.average(throughput_list))
#         scale_factor_vector[jid, :] = scale_factor
#
#     sorted_jids = np.argsort(normalized_eff_thru)
#     obj_coeff_vector = np.power(0.9, np.arange(num_jobs))
#     multi_coeff_obj = np.empty(num_jobs)
#     multi_coeff_obj[sorted_jids] = obj_coeff_vector
#
#     effective_throughput = cp.sum(cp.multiply(throughput_coeff, allocation_vars), axis=1)
#     objective = cp.Maximize(cp.sum(cp.multiply(multi_coeff_obj, effective_throughput)))
#     constraints = [allocation_vars >= 0,
#                    cp.sum(cp.multiply(scale_factor_vector, allocation_vars), axis=0) <= gpu_cap,
#                    cp.sum(allocation_vars, axis=1) <= 1,
#                    effective_throughput >= lb_rate]
#
#     model = cp.Problem(objective, constraints)
#     result = model.solve(solver=cp.GUROBI)
#     if break_down:
#         time_model = (datetime.now() - st_time_model).total_seconds()
#         return allocation_vars.value, time_model
#     return allocation_vars.value

    

