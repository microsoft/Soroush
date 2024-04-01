from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix

from scripts.problem import Problem
from gavel.scheduler.job_id_pair import JobIdPair
from alg import waterfilling_utils


def get_rates(problem: Problem, num_iter_approx_water, num_iter_adapt_water, normalized_throughput_coeff=None,
              throughput_coeff=None, scale_factor_vector=None, priority_vector=None, priority_aware=False,
              throughput_aware=False, break_down=False, return_matrix=False, biased_toward_lower_norm_eff_thru=False,
              biased_alpha=None):
    """  A fast heuristic to compute an approximate max-min fair solution without any optimizations.

    Args:
        problem: a description of a cluster scheduling problem  (e.g., available GPUs).
        num_iter_approx_water: number of iterations for the underlying approx waterfiller
            (see approx_waterfiller for more details).
        num_iter_adapt_water: number of iterations for adaptive waterfiller.

    Returns:
        job_id_to_job_rate_mapping: a mapping from job id to its assignment from each GPU.
        dur: time to find the allocation.
    """
    assert num_iter_approx_water == 1
    st_time = datetime.now()
    if throughput_coeff is None:
        output_v = waterfilling_utils.get_vectorized_characteristics(problem)
        normalized_throughput_coeff, throughput_coeff, scale_factor_vector, priority_vector = output_v

    list_jobs = problem.sparse_job_list
    num_gpu_types = len(problem.gpu_list)
    num_tokens = num_gpu_types

    bias_allocation_weight = None
    if biased_toward_lower_norm_eff_thru:
        bias_allocation_weight = np.ones(len(list_jobs))

    output = waterfilling_utils.get_routing_matrix(problem, scale_factor_vector=scale_factor_vector,
                                                   priority_aware=False, throughput_aware=False, return_details=True)
    scale_matrix, row_list, column_list, capacity_vector, jid_to_sub_jid_mapping, num_rows, num_sub_jobs = output
    num_jobs = num_sub_jobs // num_gpu_types
    split_ratios = waterfilling_utils.get_initial_split_ratio_matrix(list_jobs, problem.job_id_to_initial_split_ratio,
                                                                     num_jobs, num_gpu_types)
    one_jobs = np.ones(shape=num_sub_jobs)
    per_job_allocation = np.empty(num_sub_jobs)

    if break_down:
        finding_split_ratios_dur = (datetime.now() - st_time).total_seconds()
        routing_matrix_dur = 0
        computation_dur = 0
        updating_split_ratios_dur = 0

    for iter_bet_no in range(num_iter_adapt_water):
        if break_down:
            checkpoint = datetime.now()

        scale_matrix, weight_matrix = _get_scale_matrix(scale_factor_vector, priority_vector, row_list, column_list, num_tokens,
                                                        num_rows, num_sub_jobs, num_jobs, num_sub_jobs, num_gpu_types,
                                                        split_ratios, priority_aware=priority_aware,
                                                        throughput_aware=throughput_aware,
                                                        biased_toward_lower_norm_eff_thru=biased_toward_lower_norm_eff_thru,
                                                        bias_allocation_weight=bias_allocation_weight,
                                                        bias_alpha=biased_alpha)
        if break_down:
            routing_matrix_dur += (datetime.now() - checkpoint).total_seconds()
            checkpoint = datetime.now()

        per_job_allocation.fill(1)
        num_jobs_per_link = scale_matrix @ one_jobs
        current_fair_share = capacity_vector / num_jobs_per_link
        sorted_idx = np.argsort(current_fair_share)
        idx = 0
        num_iter = sorted_idx.shape[0]
        while idx < num_iter:
            gid = sorted_idx[idx]
            start, stop = scale_matrix.indptr[gid], scale_matrix.indptr[gid + 1]
            non_zero_indices = scale_matrix.indices[start:stop]
            _apply_congestion(scale_matrix.data[start:stop], per_job_allocation, non_zero_indices,
                              capacity_vector[gid], update_rate=True)
            idx += 1

        final_job_allocation = per_job_allocation * weight_matrix
        if break_down:
            computation_dur += (datetime.now() - checkpoint).total_seconds()
            checkpoint = datetime.now()

        if iter_bet_no == num_iter_adapt_water - 1:
            break

        throughput = np.reshape(final_job_allocation, (num_jobs, num_gpu_types))
        total_allocation = np.add.reduce(throughput * throughput_coeff, axis=1)
        if biased_toward_lower_norm_eff_thru:
            bias_allocation_weight = total_allocation * normalized_throughput_coeff
        split_ratios = throughput * throughput_coeff / total_allocation.reshape(num_jobs, -1)

        if break_down:
            updating_split_ratios_dur += (datetime.now() - checkpoint).total_seconds()

    if break_down:
        dur = (finding_split_ratios_dur, routing_matrix_dur, computation_dur, updating_split_ratios_dur)
    else:
        dur = (datetime.now() - st_time).total_seconds()

    if return_matrix:
        output = final_job_allocation.reshape(num_jobs, num_gpu_types)
        return output, dur

    job_id_to_job_rate_mapping = defaultdict(dict)
    for jid, _ in problem.sparse_job_list:
        sub_jid_list = jid_to_sub_jid_mapping[jid]
        for gid, gpu in enumerate(problem.gpu_list):
            job_id_to_job_rate_mapping[JobIdPair(jid, None)][gpu] = final_job_allocation[sub_jid_list][gid]
    print(f"approx-waterfilling: {dur}")
    return job_id_to_job_rate_mapping, dur


def _get_scale_matrix(scale_factor_vector, priority_vector, row_list, column_list, num_tokens, num_rows, num_cols,
                      num_jobs, num_sub_jobs, num_gpus, split_ratio_mapping, priority_aware=False, throughput_aware=False,
                      biased_toward_lower_norm_eff_thru=False, bias_allocation_weight=None, bias_alpha=None):
    assert not throughput_aware

    bias_coeff = 1
    if priority_aware:
        bias_coeff *= priority_vector.flatten()
    if biased_toward_lower_norm_eff_thru:
        max_allocation = np.average(bias_allocation_weight)
        bias_coeff *= np.repeat(0.000001 + np.power(bias_alpha, bias_allocation_weight / max_allocation), repeats=num_gpus)

    weight_matrix = split_ratio_mapping.flatten() * bias_coeff
    scale_data_list_1 = weight_matrix
    weight_matrix = weight_matrix / scale_factor_vector.flatten()
    scale_data_list_2 = weight_matrix
    scale_data_list = np.concatenate((scale_data_list_1, scale_data_list_2))

    scale_matrix = csr_matrix((scale_data_list, (row_list, column_list)),
                              shape=(num_rows, num_cols))
    return scale_matrix, weight_matrix


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
