from collections import defaultdict
from datetime import datetime

import numpy as np
import cvxpy as cp

from alg import waterfilling_utils
from scripts.problem import Problem
from gavel.scheduler.job_id_pair import JobIdPair


def get_rates(alpha, U, problem: Problem, min_epsilon, break_down=False):
    st_time = datetime.now()

    job_details = problem.sparse_job_list
    gpu_cap = problem.capacity_vector
    num_jobs = len(job_details)
    num_gpus = len(gpu_cap)
    output = waterfilling_utils.get_vectorized_characteristics(problem)
    norm_throughput_coefficient, throughput_coefficient, scale_factor_vector, priority_vector = output
    final_throughput_coefficient = norm_throughput_coefficient.reshape(num_jobs, 1) * throughput_coefficient

    max_demand = np.max(final_throughput_coefficient)
    U = max(U, np.min(np.add.reduce(final_throughput_coefficient, axis=1) / num_jobs))
    T = max(1, int(np.ceil(np.log(max_demand / U) / np.log(alpha))))

    bounds = U * np.power(alpha, np.arange(T + 1))
    epsilon = np.power(min_epsilon, 1.0 / T)
    multiply_coefficient = np.power(epsilon, np.arange(T + 1))

    allocation_vars = cp.Variable((num_jobs, num_gpus))
    flow_bin_vars = cp.Variable((num_jobs, T + 1))

    objective = cp.Maximize(cp.sum(flow_bin_vars @ multiply_coefficient))

    normalized_effective_throughput = cp.sum(cp.multiply(final_throughput_coefficient, allocation_vars), axis=1)
    constraints = [
        flow_bin_vars >= 0,
        allocation_vars >= 0,
        cp.sum(cp.multiply(scale_factor_vector, allocation_vars), axis=0) <= gpu_cap,
        cp.sum(allocation_vars, axis=1) <= 1,
        normalized_effective_throughput == cp.sum(flow_bin_vars, axis=1),
        flow_bin_vars[:, 1:] <= np.tile(bounds[1:] - bounds[:-1], (num_jobs, 1)),
        flow_bin_vars[:, 0] <= bounds[0],
    ]

    model = cp.Problem(objective, constraints)
    if break_down:
        checkpoint1 = datetime.now()
    model.solve(
        solver=cp.GUROBI,
        Method=2,
        QCPDual=0,
        # Crossover=0
    )
    # Crossover=0, BarConvTol=1e-4
    dur = (datetime.now() - st_time).total_seconds()
    if break_down:
        curr_time = datetime.now()
        time_model = (checkpoint1 - st_time).total_seconds()
        time_solve = (curr_time - checkpoint1).total_seconds()
        return allocation_vars.value, (time_model, time_solve)

    job_id_to_job_rate_mapping = defaultdict(dict)
    final_job_allocation_matrix = allocation_vars.value
    for jid, _ in job_details:
        for gid, gpu in enumerate(problem.gpu_list):
            job_id_to_job_rate_mapping[JobIdPair(jid, None)][gpu] = final_job_allocation_matrix[jid, gid]
    return job_id_to_job_rate_mapping, dur
