from collections import defaultdict
from datetime import datetime

import numpy as np

from alg import waterfilling_utils
from utilities import constants
from scripts.problem import Problem
from gavel.scheduler.job_id_pair import JobIdPair


def get_rates(problem: Problem, num_iter, priority_aware=False, throughput_aware=False,
              break_down=False, return_matrix=False):
    st_time = datetime.now()
    output = waterfilling_utils.get_routing_matrix(problem, priority_aware=priority_aware, throughput_aware=throughput_aware)
    # print(f"creating traffic;", (datetime.now() - st_time).total_seconds())
    scale_matrix, weight_matrix, capacity_vector, jid_to_sub_jid_mapping, num_jobs = output
    if break_down:
        checkpoint_1_time = datetime.now()

    per_job_allocation = np.full(num_jobs, 1.0)
    final_job_allocation = np.empty(shape=num_jobs)
    np_job_list = np.arange(num_jobs)

    num_scaled_jobs_per_gpu = scale_matrix @ np.ones(shape=num_jobs)
    unfreezed_jobs = np.ones(shape=num_jobs)
    for i in range(num_iter - 1):
        current_fair_share = capacity_vector / num_scaled_jobs_per_gpu
        s_sparse = scale_matrix @ scale_matrix.transpose()
        min_gpu = waterfilling_utils.min_neighbor_fair_share(s_sparse, current_fair_share)
        bottlenecked_jobs = np.ones_like(per_job_allocation)
        gpu_mask = np.full(current_fair_share.shape, fill_value=True)
        for gid, rate in min_gpu:
            gpu_mask[gid] = False
            desired_jobs = scale_matrix.getrow(gid).indices
            bottlenecked_jobs[desired_jobs] = 0
            per_job_allocation[desired_jobs] = rate
            final_job_allocation[np_job_list[desired_jobs]] = rate
            unfreezed_jobs[np_job_list[desired_jobs]] = 0

        capacity_vector = np.compress(gpu_mask, capacity_vector)
        scale_matrix = scale_matrix[gpu_mask]
        capacity_vector -= scale_matrix @ (per_job_allocation * (1 - bottlenecked_jobs))
        scale_matrix = scale_matrix[:, bottlenecked_jobs > 0]
        np_job_list = np.compress(bottlenecked_jobs, np_job_list)
        per_job_allocation = np.compress(bottlenecked_jobs, per_job_allocation)

        num_scaled_jobs_per_gpu = scale_matrix @ np.ones(shape=per_job_allocation.shape[0])
        # while num_scaled_jobs_per_gpu.shape[0] and np.any(num_scaled_jobs_per_gpu == 0):
        #     mask = (num_scaled_jobs_per_gpu > 0)
        #     routing_matrix = routing_matrix[mask]
        #     scale_matrix = scale_matrix[mask]
        #     capacity_vector = np.compress(mask, capacity_vector)
        #     num_scaled_jobs_per_gpu = scale_matrix @ np.ones(shape=per_job_allocation.shape[0])

    current_fair_share = capacity_vector / num_scaled_jobs_per_gpu
    sorted_idx = np.argsort(current_fair_share)
    idx = 0
    num_iter = sorted_idx.shape[0]
    while idx < num_iter:
        gid = sorted_idx[idx]
        start, stop = scale_matrix.indptr[gid],  scale_matrix.indptr[gid + 1]
        non_zero_indices = scale_matrix.indices[start:stop]
        _apply_congestion(scale_matrix.data[start:stop], per_job_allocation, non_zero_indices,
                          capacity_vector[gid], update_rate=True)
        idx += 1

    final_job_allocation[unfreezed_jobs > 0] = per_job_allocation
    final_job_allocation = final_job_allocation * weight_matrix

    if break_down:
        checkpoint_2_time = datetime.now()
        dur = ((checkpoint_1_time - st_time).total_seconds(), (checkpoint_2_time - checkpoint_1_time).total_seconds())
    else:
        dur = (datetime.now() - st_time).total_seconds()

    if return_matrix:
        num_gpus = len(problem.gpu_list)
        output = final_job_allocation.reshape(num_jobs // num_gpus, num_gpus)
        return output, dur

    job_id_to_job_rate_mapping = defaultdict(dict)
    for jid, _ in problem.sparse_job_list:
        sub_jid_list = jid_to_sub_jid_mapping[jid]
        for gid, gpu in enumerate(problem.gpu_list):
            job_id_to_job_rate_mapping[JobIdPair(jid, None)][gpu] = final_job_allocation[sub_jid_list][gid]
    print(f"approx-waterfilling;", dur)
    return job_id_to_job_rate_mapping, dur


def _apply_congestion(scale_matrix, job_allocation, non_zeros_jids, gpu_cap, update_rate):
    job_allocation_on_gpus = job_allocation[non_zeros_jids]
    scaled_allocation = job_allocation_on_gpus * scale_matrix
    if np.add.reduce(scaled_allocation) <= gpu_cap or job_allocation_on_gpus.shape[0] == 0:
        return np.inf
    mask = np.arange(job_allocation_on_gpus.shape[0])
    while mask.shape[0]:
        num_scaled_jobs = np.add.reduce(scale_matrix[mask])
        fair_share = gpu_cap / num_scaled_jobs
        under_flows = (job_allocation_on_gpus[mask] >= fair_share)
        if np.logical_and.reduce(under_flows):
            job_allocation_on_gpus[mask] = fair_share
            break
        else:
            gpu_cap -= scaled_allocation[mask] @ (1 - under_flows)
            mask = np.compress(under_flows, mask)
    if update_rate:
        job_allocation[non_zeros_jids] = job_allocation_on_gpus
    return fair_share


