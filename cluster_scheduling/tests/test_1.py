import os, sys
import copy
from collections import defaultdict

import numpy as np
import random
import time
from matplotlib import pyplot as plt
import seaborn as sn

sys.path.append(os.path.abspath(os.path.join('../gavel/scheduler')))
sys.path.append(os.path.abspath(os.path.join('../')))
from job_id_pair import JobIdPair
import utils as gavel_utils
from alg import gavel_max_min_fairness_waterfilling
from alg import approx_waterfiller
from alg import adapt_waterfiller
from alg import equi_depth_binner
from alg import sorting_network_exact
from scripts.problem import Problem
from utilities import constants


np.set_printoptions(precision=3, suppress=True)


def create_problem_instance(num_jobs, cluster_spec, seed, ss_share, introduce_skew):
    oracle_throughputs = gavel_utils.read_all_throughputs_json_v2("../gavel/scheduler/simulation_throughputs.json")
    rng = random.Random()
    rng.seed(seed)
    jobs = {}
    throughputs = {}
    scale_factors = {}
    priority_mapping = {}
    for i in range(num_jobs):
        job_id = JobIdPair(i, None)
        job = gavel_utils.generate_job(throughputs=oracle_throughputs, rng=rng, job_id=job_id,
                                       generate_multi_gpu_jobs=True)
                                       # scale_factor_generator_func=lambda x: x.choice([1, 2, 4, 8]))
        jobs[job_id[0]] = job
        job_type_key = (job.job_type, job.scale_factor)
        throughputs[job_id] = {}
        for worker_type in cluster_spec:
            throughputs[job_id][worker_type] = \
                oracle_throughputs[worker_type][job_type_key]['null']
        # scale_factors[job_id] = np.random.choice([1, 2, 4, 8], 1)[0]
        scale_factors[job_id] = job.scale_factor
        if introduce_skew:
            priority_mapping[job_id] = (i % 4) + 1.0
        else:
            priority_mapping[job_id] = 1.0
        # job._priority_weight = priority_mapping[job_id]

    if ss_share:
        for i in range(num_jobs):
            job_type_key = (jobs[i].job_type, jobs[i].scale_factor)
            for j in range(num_jobs):
                if i < j and jobs[i].scale_factor == jobs[j].scale_factor:
                    other_job_type_key = \
                        (jobs[j].job_type, jobs[j].scale_factor)
                    throughputs[JobIdPair(i, j)] = {}
                    for worker_type in cluster_spec:
                        throughputs[JobIdPair(i, j)][worker_type] = \
                            oracle_throughputs[worker_type][job_type_key][other_job_type_key]
                    # scale_factors[jobs[i].job_id, jobs[j].job_id] = 1

    problem = Problem(cluster_spec, job_dict=jobs, oracle_throughputs=oracle_throughputs, priority_mapping=priority_mapping)
    # print(len(throughputs))
    # print("scale_factor==========", scale_factors)
    return throughputs, scale_factors, priority_mapping, problem


def sweep(all_num_jobs, num_trials, ss_share, introduce_skew):
    all_runtimes = {constants.GAVEL: {}, constants.APPROX.format(1): {}, constants.APPROX.format(1) + "-prio-thru-aware": {},
                    constants.APPROX_BET.format(1, 10): {}, constants.APPROX_BET.format(1, 10) + "-prio-thru-aware": {},
                    constants.APPROX_MCF.format(1) + "-prio-thru-aware": {}, constants.APPROX_BET_MCF.format(1, 10) + "-prio-thru-aware": {},
                    constants.APPROX_BET_MCF.format(1, 10) + "-prio-thru-aware-biased": {}, constants.GAVEL + "-wo-waterfilling": {},}
                    # constants.OPT_SORTING_NETWORK: {}}
    all_effective_throughputs = {}
    for key in all_runtimes:
        all_effective_throughputs[key] = {}

    for num_jobs in all_num_jobs:
        for policy in all_runtimes:
            all_runtimes[policy][num_jobs] = []
            all_effective_throughputs[policy][num_jobs] = []
        cluster_spec = {
            'v100': max(num_jobs // 4, 1),
            'p100': max(num_jobs // 4, 1),
            'k80': max(num_jobs // 4, 1),
        }
        for i in range(num_trials):
            throughputs, scale_factors, priority_mapping, problem = create_problem_instance(num_jobs, cluster_spec, seed=i,
                                                                                            ss_share=ss_share, introduce_skew=introduce_skew)
            print("======================", f"{constants.GAVEL}-w-waterfilling")
            allocation, runtime = gavel_max_min_fairness_waterfilling.get_allocation(throughputs, scale_factors, priority_mapping,
                                                                                     cluster_spec, ss_share,
                                                                                     waterfilling_enabled=True)
            all_runtimes[constants.GAVEL][num_jobs].append(runtime)
            effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            all_effective_throughputs[constants.GAVEL][num_jobs].append(effective_throughputs)

            print("======================", f"{constants.GAVEL}-wo-waterfilling")
            allocation, runtime = gavel_max_min_fairness_waterfilling.get_allocation(throughputs, scale_factors, priority_mapping, cluster_spec,
                                                                                     ss_share, waterfilling_enabled=False)
            all_runtimes[constants.GAVEL + "-wo-waterfilling"][num_jobs].append(runtime)
            effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            all_effective_throughputs[constants.GAVEL + "-wo-waterfilling"][num_jobs].append(effective_throughputs)

            print("======================", f"{constants.APPROX.format(1)}")
            allocation, runtime = approx_waterfiller.get_rates(problem, num_iter=1, priority_aware=False, throughput_aware=False)
            effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            all_runtimes[constants.APPROX.format(1)][num_jobs].append(runtime)
            all_effective_throughputs[constants.APPROX.format(1)][num_jobs].append(effective_throughputs)

            print("======================", f"{constants.APPROX.format(1)}-prio-thru-aware")
            allocation, runtime = approx_waterfiller.get_rates(problem, num_iter=1, priority_aware=True, throughput_aware=True)
            effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            all_runtimes[constants.APPROX.format(1) + "-prio-thru-aware"][num_jobs].append(runtime)
            all_effective_throughputs[constants.APPROX.format(1) + "-prio-thru-aware"][num_jobs].append(effective_throughputs)

            print("======================", f"{constants.APPROX_BET.format(1, 10)}-isolation")
            allocation, runtime = adapt_waterfiller.get_rates(problem, num_iter_approx_water=1, num_iter_adapt_water=10)
            effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            all_runtimes[constants.APPROX_BET.format(1, 10)][num_jobs].append(runtime)
            all_effective_throughputs[constants.APPROX_BET.format(1, 10)][num_jobs].append(effective_throughputs)

            print("======================", f"{constants.APPROX_BET.format(1, 10)}-prio-thru-aware")
            allocation, runtime = adapt_waterfiller.get_rates(problem, num_iter_approx_water=1, num_iter_adapt_water=10,
                                                             biased_toward_lower_norm_eff_thru=True,
                                                             biased_alpha=0.5,
                                                             priority_aware=True, throughput_aware=False)
            effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            all_runtimes[constants.APPROX_BET.format(1, 10) + "-prio-thru-aware"][num_jobs].append(runtime)
            all_effective_throughputs[constants.APPROX_BET.format(1, 10) + "-prio-thru-aware"][num_jobs].append(effective_throughputs)

            # print("======================", f"{constants.APPROX_MCF.format(1)}-prio-thru-aware")
            # allocation, runtime = approx_water_plus_mcf.get_rates(problem, num_approx_iter=1, priority_aware=True,
            #                                                       epsilon=0.995, k=1, alpha=0, beta=0.01, throughput_aware=True)
            # effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            # all_runtimes[constants.APPROX_MCF.format(1) + "-prio-thru-aware"][num_jobs].append(runtime)
            # all_effective_throughputs[constants.APPROX_MCF.format(1) + "-prio-thru-aware"][num_jobs].append(effective_throughputs)

            print("======================", f"{constants.APPROX_BET_MCF.format(1, 10)}-prio-thru-aware")
            # allocation, runtime = equi_depth_binner.get_rates(problem, num_iter_approx_water=1, num_iter_bet=10,
            #                                                           priority_aware=True, epsilon=0.995, k=1, alpha=0,
            #                                                           beta=0.01, throughput_aware=False)
            
            allocation, runtime = equi_depth_binner.get_rates(problem, num_iter_approx_water=1, num_iter_bet=2,
                                                              priority_aware=True, min_epsilon=1e-6, k=1, alpha=0,
                                                              min_beta=1e-4, num_bins=5, throughput_aware=False)
            # allocation, runtime = approx_water_bet_plus_mcf.get_rates(problem, num_iter_approx_water=1, num_iter_bet=2,
            #                                                           priority_aware=True, epsilon=0.995, k=1, alpha=0,
            #                                                           beta=0.99, throughput_aware=False,
            #                                                           biased_toward_lower_norm_eff_thru=True,
            #                                                           biased_approx_bet_alpha=0.5)
            effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            all_runtimes[constants.APPROX_BET_MCF.format(1, 10) + "-prio-thru-aware"][num_jobs].append(runtime)
            all_effective_throughputs[constants.APPROX_BET_MCF.format(1, 10) + "-prio-thru-aware"][num_jobs].append(effective_throughputs)

            print("======================", f"{constants.APPROX_BET_MCF.format(1, 10)}-prio-thru-aware-biased")
            # allocation, runtime = equi_depth_binner.get_rates(problem, num_iter_approx_water=1, num_iter_bet=2,
            #                                                           epsilon=0.995, k=1, alpha=0, beta=0.01,
            #                                                           priority_aware=True, throughput_aware=False,
            #                                                           biased_toward_lower_norm_eff_thru=True,
            #                                                           biased_approx_bet_alpha=0.5, break_down=False)
            
            allocation, runtime = equi_depth_binner.get_rates(problem, num_iter_approx_water=1, num_iter_bet=2,
                                                              min_epsilon=1e-6, k=1, alpha=0, min_beta=0, num_bins=5,
                                                              priority_aware=True, throughput_aware=False,
                                                              biased_toward_lower_norm_eff_thru=True,
                                                              biased_approx_bet_alpha=0.5, break_down=False)
            effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            all_runtimes[constants.APPROX_BET_MCF.format(1, 10) + "-prio-thru-aware-biased"][num_jobs].append(runtime)
            all_effective_throughputs[constants.APPROX_BET_MCF.format(1, 10) + "-prio-thru-aware-biased"][num_jobs].append(effective_throughputs)

            # print("=====================", f"{constants.OPT_SORTING_NETWORK}")
            # allocation, runtime = optimization_plus_sorting_networks.get_rates(problem, break_down=True)
            # effective_throughputs = compute_effective_throughput(problem, allocation, throughputs)
            # all_runtimes[constants.OPT_SORTING_NETWORK][num_jobs].append(runtime)
            # all_effective_throughputs[constants.OPT_SORTING_NETWORK][num_jobs].append(effective_throughputs)

    return all_runtimes, all_effective_throughputs


def compute_effective_throughput(problem:Problem, allocation, throughputs):
    effective_throughputs = {}
    all_allocation = {}
    gpu_allocation = defaultdict(int)
    for job_id in allocation:
        for single_job_id in job_id.singletons():
            if single_job_id not in effective_throughputs:
                effective_throughputs[single_job_id] = 0.0
                all_allocation[single_job_id] = 0.0
        for worker_type in allocation[job_id]:
            gpu_allocation[worker_type] += allocation[job_id][worker_type] * problem.job_id_to_scale_factor[job_id]
            if gpu_allocation[worker_type] > problem.cluster_spec[worker_type] + constants.O_epsilon:
                print(gpu_allocation[worker_type])
            assert gpu_allocation[worker_type] <= problem.cluster_spec[worker_type] + constants.O_epsilon + 0.001
            for jidx, single_job_id in enumerate(job_id.singletons()):
                all_allocation[single_job_id] += allocation[job_id][worker_type]
                if job_id.is_pair():
                    effective_throughputs[single_job_id] += (
                            allocation[job_id][worker_type] *
                            throughputs[job_id][worker_type][jidx])
                else:
                    effective_throughputs[single_job_id] += (
                            allocation[job_id][worker_type] *
                            throughputs[job_id][worker_type])
                if all_allocation[single_job_id] > 1 + constants.O_epsilon:
                    print(single_job_id, all_allocation[single_job_id])
                assert all_allocation[single_job_id] <= 1 + constants.O_epsilon + 0.001
    return effective_throughputs


def get_runtimes_and_effective_throughputs(num_jobs_list, ss_share, introduce_skew):
    all_runtimes, all_effective_throughputs = sweep(num_jobs_list, num_trials=1, ss_share=ss_share, introduce_skew=introduce_skew)
    return all_runtimes, all_effective_throughputs


if __name__ == '__main__':
    runtimes, all_effective_throughputs = \
        get_runtimes_and_effective_throughputs([8192], ss_share=False, introduce_skew=True)
    baseline_policy = constants.GAVEL
    fairness_cdf = defaultdict(dict)
    total_effect = dict()
    fairness_geometric_mean = dict()
    for policy, num_jobs_eff_thru_mapping in all_effective_throughputs.items():
        total_effect[policy] = {}
        fairness_geometric_mean[policy] = {}
        for num_jobs, sample_effective_throughput_mapping in num_jobs_eff_thru_mapping.items():
            total_effect[policy][num_jobs] = {}
            fairness_geometric_mean[policy][num_jobs] = {}
            for sidx, effective_throughput_mapping in enumerate(sample_effective_throughput_mapping):
                total_effect[policy][num_jobs][sidx] = 0
                fairness_geometric_mean[policy][num_jobs][sidx] = 0
                effective_thru = effective_throughput_mapping
                fairness_geometric_mean_num = 0
                baseline_effective_thru = all_effective_throughputs[baseline_policy][num_jobs][sidx]
                fairness_cdf[num_jobs][f"{policy}-{sidx}"] = []
                for jid in effective_thru:
                    total_effect[policy][num_jobs][sidx] += effective_thru[jid]
                    fair_num = np.maximum(0.1, effective_thru[jid]) / np.maximum(0.1, baseline_effective_thru[jid])
                    fair_num_idx = np.minimum(fair_num, 1.0/fair_num)
                    # if fair_num < 0.95:
                    #     print("done")
                    # if fair_num > 2:
                    #     print("done")
                    fairness_geometric_mean_num += np.log10(fair_num_idx)
                    fairness_cdf[num_jobs][f"{policy}-{sidx}"].append(fair_num)
                fairness_geometric_mean_num /= len(effective_thru)
                fairness_geometric_mean[policy][num_jobs][sidx] = np.power(10, fairness_geometric_mean_num)

    print(runtimes)
    print(total_effect)
    print(fairness_geometric_mean)
    # for num_jobs in fairness_cdf:
    #     plt.figure()
    #     for policy in fairness_cdf[num_jobs]:
    #         sn.ecdfplot(fairness_cdf[num_jobs][policy], label=policy)
    #     plt.ylabel("Fraction of Jobs")
    #     plt.xlabel("Throughput, relative to Gavel")
    #     plt.legend()
    #     plt.xlim([0, 3])
    #     plt.show()
    # print(all_effective_throughputs)