import numpy as np

from datetime import datetime
from collections import defaultdict

from alg import approx_water_bet
from alg import approx_water_plus_mcf
from alg import waterfilling_utils
from scripts.problem import Problem
from gavel.scheduler.job_id_pair import JobIdPair


def get_rates(problem: Problem, num_iter_approx_water, num_iter_bet, epsilon, k, alpha, beta, priority_aware=False,
              throughput_aware=False, break_down=False, biased_toward_lower_norm_eff_thru=False, biased_approx_bet_alpha=None):
    assert (not biased_toward_lower_norm_eff_thru) or (biased_approx_bet_alpha is not None)
    st_time = datetime.now()
    output = waterfilling_utils.get_vectorized_characteristics(problem)
    normalized_throughput_coeff, throughput_coeff, scale_factor_vector, priority_vector = output
    if break_down:
        checkpoint = datetime.now()
    allocation_matrix, approx_dur = approx_water_bet.get_rates(problem, num_iter_approx_water=num_iter_approx_water,
                                                               num_iter_bet=num_iter_bet,
                                                               normalized_throughput_coeff=normalized_throughput_coeff,
                                                               throughput_coeff=throughput_coeff,
                                                               scale_factor_vector=scale_factor_vector,
                                                               priority_vector=priority_vector,
                                                               priority_aware=priority_aware,
                                                               throughput_aware=throughput_aware, break_down=break_down,
                                                               return_matrix=True,
                                                               biased_toward_lower_norm_eff_thru=biased_toward_lower_norm_eff_thru,
                                                               biased_alpha=biased_approx_bet_alpha)

    job_details = problem.sparse_job_list
    output = approx_water_plus_mcf.compute_throughput_given_jobs(job_details, allocation_matrix, normalized_throughput_coeff,
                                                                 throughput_coeff, scale_factor_vector, problem.capacity_vector,
                                                                 epsilon=epsilon, k=k, alpha=alpha, beta=beta, break_down=break_down)
    dur = (datetime.now() - st_time).total_seconds()
    if break_down:
        final_job_allocation_matrix, detailed_dur = output
        dur = ((checkpoint - st_time).total_seconds(), approx_dur, detailed_dur)
    else:
        final_job_allocation_matrix = output

    job_id_to_job_rate_mapping = defaultdict(dict)
    for jid, _ in job_details:
        for gid, gpu in enumerate(problem.gpu_list):
            job_id_to_job_rate_mapping[JobIdPair(jid, None)][gpu] = final_job_allocation_matrix[jid, gid]

    # Checking the final solution
    throughput_coeff = np.empty_like(allocation_matrix)
    for jid, (priority_weight, scale_factor, throughput_list) in job_details:
        throughput_coeff[jid] = throughput_list
    final_thru = np.sum(np.multiply(throughput_coeff, final_job_allocation_matrix), axis=1)
    initial_thru = np.sum(np.multiply(throughput_coeff, allocation_matrix), axis=1)
    # assert np.all(final_thru >= initial_thru - constants.O_epsilon)
    print(np.max(initial_thru), np.min(initial_thru))
    print(np.max(final_thru), np.min(final_thru))

    return job_id_to_job_rate_mapping, dur
