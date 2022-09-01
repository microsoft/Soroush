import numpy as np
from scipy.sparse import csr_matrix

from scripts.problem import Problem
from utilities import constants


def get_routing_matrix(problem: Problem, scale_factor_vector=None, priority_aware=False, throughput_aware=False,
                       return_details=False):
    if priority_aware or throughput_aware:
        assert not return_details
        return get_heterogeneous_routing_matrix(problem, priority_aware, throughput_aware)

    assert not priority_aware and not throughput_aware
    return get_homogeneous_routing_matrix(problem, scale_factor_vector, return_details)


def get_homogeneous_routing_matrix(problem: Problem, scale_factor_vector=None, return_details=False):

    if scale_factor_vector is None:
        _, _, scale_factor_vector, _ = get_vectorized_characteristics(problem)
    list_jobs = problem.sparse_job_list
    gpu_cap_vector = problem.capacity_vector
    num_jobs = len(list_jobs)
    num_gpu_types = len(gpu_cap_vector)
    sub_jid = num_jobs * num_gpu_types
    max_gpu_id = num_gpu_types + num_jobs

    row_list_1 = np.tile(np.arange(num_gpu_types, dtype=np.int32), reps=num_jobs)
    column_list_1 = np.arange(sub_jid)
    scale_list_1 = scale_factor_vector.flatten()

    row_list_2 = np.repeat(np.arange(num_gpu_types, max_gpu_id), repeats=num_gpu_types)
    column_list_2 = np.arange(sub_jid)
    scale_list_2 = np.full(sub_jid, fill_value=1)

    row_list = np.concatenate((row_list_1, row_list_2))
    column_list = np.concatenate((column_list_1, column_list_2))
    scale_factor_data_list = np.concatenate((scale_list_1, scale_list_2))
    scale_matrix = csr_matrix((scale_factor_data_list, (row_list, column_list)),
                              shape=(max_gpu_id, sub_jid))

    jid_to_sub_jid_mapping = np.arange(sub_jid).reshape(num_jobs, num_gpu_types)

    capacity_vector = np.ones(shape=max_gpu_id)
    capacity_vector[:num_gpu_types] = gpu_cap_vector[:]

    if return_details:
        return scale_matrix, row_list, column_list, capacity_vector, jid_to_sub_jid_mapping, \
               num_gpu_types + num_jobs, sub_jid

    weight_matrix = np.ones(sub_jid)
    return scale_matrix, weight_matrix, capacity_vector, jid_to_sub_jid_mapping, sub_jid


def get_heterogeneous_routing_matrix(problem: Problem, priority_aware=False, throughput_aware=False):
    output_v = get_vectorized_characteristics(problem)
    normalized_throughput_coeff, throughput_coeff, scale_factor_vector, priority_vector = output_v
    weight_matrix = np.ones_like(throughput_coeff.flatten())
    num_jobs = len(problem.sparse_job_list)

    output = get_routing_matrix(problem, scale_factor_vector=scale_factor_vector,
                                priority_aware=False, throughput_aware=False, return_details=True)
    scale_matrix, row_list, column_list, capacity_vector, jid_to_sub_jid_mapping, num_rows, num_sub_jobs = output

    if priority_aware:
        weight_matrix *= priority_vector.flatten()

    if throughput_aware:
        coeff = np.power(constants.SPLIT_CONST, throughput_coeff) * (throughput_coeff >= constants.O_epsilon)
        coeff /= np.add.reduce(coeff, axis=1).reshape(num_jobs, -1)
        np.nan_to_num(coeff, posinf=0, neginf=0, copy=False)
        weight_matrix *= coeff.flatten()

    scale_data_list_1 = weight_matrix
    weight_matrix = weight_matrix / scale_factor_vector.flatten()
    scale_data_list_2 = weight_matrix
    scale_data_list = np.concatenate((scale_data_list_1, scale_data_list_2))

    scale_matrix = csr_matrix((scale_data_list, (row_list, column_list)),
                              shape=(num_rows, num_sub_jobs))
    return scale_matrix, weight_matrix, capacity_vector, jid_to_sub_jid_mapping, num_sub_jobs

def get_heterogeneous_routing_matrix_1(problem: Problem, priority_aware=False, throughput_aware=False):
    list_jobs = problem.sparse_job_list
    gpu_cap_vector = problem.capacity_vector
    jid_to_sub_jid_mapping = dict()
    num_jobs = len(list_jobs)
    num_gpu_types = len(gpu_cap_vector)
    capacity_vector = np.empty(shape=(num_jobs + num_gpu_types))
    capacity_vector[:num_gpu_types] = gpu_cap_vector[:]
    sub_jid = 0
    np_idx = 0
    max_gpu_id = num_gpu_types
    row_list = np.empty(num_gpu_types * 2 * num_jobs)
    column_list = np.empty(num_gpu_types * 2 * num_jobs)
    scale_factor_data_list = np.empty(num_gpu_types * 2 * num_jobs)
    weight_matrix = np.empty(num_gpu_types * num_jobs)
    for jid, (priority_weight, scale_factor, throughput_list) in list_jobs:
        row_list[np_idx:np_idx + num_gpu_types] = np.arange(num_gpu_types, dtype=np.int32)
        column_list[np_idx:np_idx + num_gpu_types] = np.arange(sub_jid, sub_jid + num_gpu_types)
        weight = 1
        if priority_aware:
            weight *= priority_weight
        if throughput_aware:
            coeff = np.power(constants.SPLIT_CONST, throughput_list) * (throughput_list >= constants.O_epsilon)
            coeff /= np.sum(coeff)
            weight *= coeff #* (np.sum(throughput_list) / throughput_list)
            np.nan_to_num(weight, posinf=0, neginf=0, copy=False)
        scale_factor_data_list[np_idx:np_idx + num_gpu_types] = weight
        weight /= scale_factor
        np_idx += num_gpu_types

        column_list[np_idx: np_idx + num_gpu_types] = np.arange(sub_jid, sub_jid + num_gpu_types)
        row_list[np_idx: np_idx + num_gpu_types] = max_gpu_id
        scale_factor_data_list[np_idx:np_idx + num_gpu_types] = weight
        np_idx += num_gpu_types

        jid_to_sub_jid_mapping[jid] = np.arange(sub_jid, sub_jid + num_gpu_types)
        weight_matrix[sub_jid: sub_jid + num_gpu_types] = weight
        sub_jid += num_gpu_types
        capacity_vector[max_gpu_id] = 1
        max_gpu_id += 1

    scale_matrix = csr_matrix((scale_factor_data_list, (row_list, column_list)),
                              shape=(num_gpu_types + num_jobs, sub_jid))
    return scale_matrix, weight_matrix, capacity_vector, jid_to_sub_jid_mapping, sub_jid


def min_neighbor_fair_share(s_sparse, fair_share):
    x_len = s_sparse.shape[0]
    min_link = []
    for i in range(x_len):
        start, stop = s_sparse.indptr[i], s_sparse.indptr[i + 1]
        non_zero_indices = s_sparse.indices[start:stop]
        min_fair = np.min(fair_share[non_zero_indices])
        if fair_share[i] <= min_fair + constants.O_epsilon:
            min_link.append((i, fair_share[i]))
    return min_link


def get_initial_split_ratio_matrix(list_jobs, job_split_ratio_mapping, num_jobs, num_gpus):
    split_ratios = np.empty((num_jobs, num_gpus))
    for jid, (_, _, _) in list_jobs:
        split_ratios[jid] = np.copy(job_split_ratio_mapping[jid])
    return split_ratios


def get_vectorized_characteristics(problem: Problem):
    gpu_cap = problem.capacity_vector
    job_list = problem.sparse_job_list
    num_jobs = len(job_list)
    num_gpus = len(gpu_cap)
    normalized_throughput_coeff = np.empty(num_jobs)
    throughput_coeff = np.empty((num_jobs, num_gpus))
    scale_factor_vector = np.empty((num_jobs, num_gpus))
    priority_vector = np.empty((num_jobs, num_gpus))
    for jid, (priority_weight, scale_factor, throughput_list) in job_list:
        normalized_throughput_coeff[jid] = scale_factor / (priority_weight * np.average(throughput_list, weights=gpu_cap))
        throughput_coeff[jid] = throughput_list
        scale_factor_vector[jid] = scale_factor
        priority_vector[jid] = priority_weight

    return normalized_throughput_coeff, throughput_coeff, scale_factor_vector, priority_vector
