import os
from collections import defaultdict
from glob import iglob

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from alg import approx_waterfilling
from utilities import shortest_paths, constants
from alg import swan_max_min_approx
from utilities import utils
from ncflow.lib.problem import Problem
from alg import k_waterfilling
from alg import danna_practical_max_min_fair
from alg import iewf_upward_max_min
from alg import approx_water_plus_mcf, approx_water_bet
from alg import approx_water_bet_plus_mcf


# TM_MODEL = 'poisson-high-inter'
TM_MODEL = 'gravity'

# TOPO_NAME = 'Kdl.graphml'
# TOPO_NAME = 'Cogentco.graphml'
# TOPO_NAME = 'GtsCe.graphml'
# TOPO_NAME = 'UsCarrier.graphml'
# TOPO_NAME = 'Colt.graphml'
# TOPO_NAME = 'Ion.graphml'
TOPO_NAME = 'Uninett2010.graphml'
LOG_DIR = f"./outputs/examples/{TOPO_NAME}/{TM_MODEL}"
utils.ensure_dir(LOG_DIR)
f_idx = 1


def find_topo_tm_fname(topo_name, tm_model):
    fname_list = []
    topo_fname = os.path.join('.', 'ncflow', 'topologies', 'topology-zoo', topo_name)
    for tm_fname in iglob(
            './ncflow/traffic-matrices/{}/{}*_traffic-matrix.pkl'.format(
                tm_model, topo_name)):
        vals = os.path.basename(tm_fname)[:-4].split('_')
        _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
        fname_list.append((topo_name, tm_model, scale_factor, topo_fname, tm_fname))
    return fname_list


fnames = find_topo_tm_fname(TOPO_NAME, TM_MODEL)
print(fnames)
num_paths_per_flow = 16

num_iter_bet = 10
file_name = fnames[f_idx]
problem = Problem.from_file(file_name[3], file_name[4])
utils.revise_list_commodities(problem)

output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx, k=num_paths_per_flow, number_processors=32,
                                                               base_split=0.9, split_type=constants.EXPONENTIAL_DECAY)
path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = output

danna_file_path = f"./outputs/examples/{TOPO_NAME}/{TM_MODEL}/danna_gtsc_{f_idx}.pickle"
if utils.file_exists(danna_file_path):
    danna_fid_to_total_rate = utils.read_pickle_file(danna_file_path)
else:
    danna_fid_to_total_rate, danna_dur = danna_practical_max_min_fair.get_rates(0.1, problem, path_dict, link_cap=1000.0, feasibility_grb_method=1, mcf_grb_method=1)
    utils.write_to_file_as_pickle(danna_fid_to_total_rate, LOG_DIR, danna_file_path)

# iewf_fid_to_flow_rate_mapping, iewf_dur = iewf_upward_max_min.get_rates(problem, path_edge_idx_mapping, 1000.0, num_paths_per_flow, 2)
# danna_fid_to_total_rate = utils.read_pickle_file("./outputs/danna_2.pickle")
# danna_fid_to_total_rate = utils.read_pickle_file("./outputs/danna_gtsc_5.pickle")
# danna_dur = 2074.658941

# approx_water_fid_to_flow_rate_mapping, approx_dur = approx_waterfilling.get_rates(problem, path_edge_idx_mapping,
#                                                                                   num_paths_per_flow, num_iter=1,
#                                                                                   link_cap=1000.0, break_down=False)

# problem: Problem, path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet,
#               link_cap, path_characteristics=None, path_edge_idx_mapping=None, path_path_len_mapping=None,
#               break_down=False, bias_toward_low_flow_rate=False, bias_alpha=None, return_matrix=False
approx_water_bet_fid_to_flow_rate_mapping, approx_w_bet_dur, _ = approx_water_bet.get_rates(problem,
                                                                                            path_split_ratio_mapping,
                                                                                            num_paths_per_flow,
                                                                                            num_iter_approx_water=1,
                                                                                            num_iter_bet=num_iter_bet,
                                                                                            link_cap=1000.0,
                                                                                            path_edge_idx_mapping=path_edge_idx_mapping,
                                                                                            path_path_len_mapping=path_path_len_mapping,
                                                                                            bias_toward_low_flow_rate=False,
                                                                                            bias_alpha=0.5)

# for fid, (src, dst, demand) in problem.sparse_commodity_list:
#     total_approx_1 = np.sum(approx_water_bet_fid_to_flow_rate_mapping[src, dst])
#     total_approx_2 = np.sum(approx_water_bet_fid_to_flow_rate_mapping_2[src, dst])
#     print(total_approx_1, total_approx_2)
#     assert np.abs(total_approx_1 - total_approx_2) <= constants.O_epsilon
# print("all flows are ok!")
print(file_name)
approx_water_bet_p_mcf_fid_to_flow_rate_mapping, approx_w_bet_p_mcf_dur = approx_water_bet_plus_mcf.get_rates(problem,
                                                                                                              path_dict,
                                                                                                              path_edge_idx_mapping,
                                                                                                              path_path_len_mapping,
                                                                                                              path_total_path_len_mapping,
                                                                                                              path_split_ratio_mapping,
                                                                                                              num_paths_per_flow,
                                                                                                              num_iter_approx_water=1,
                                                                                                              num_iter_bet=num_iter_bet,
                                                                                                              link_cap=1000.0,
                                                                                                              break_down=False,
                                                                                                              biased_toward_low_flow_rate=False,
                                                                                                              biased_alpha=0.5)

alpha = 2
U = 0.1

swan_file_path = f"./outputs/examples/{TOPO_NAME}/{TM_MODEL}/swan_gtsc_{f_idx}.pickle"
if utils.file_exists(swan_file_path):
    swan_fid_to_flow_rate_mapping = utils.read_pickle_file(swan_file_path)
else:
    swan_fid_to_flow_rate_mapping, swan_dur = swan_max_min_approx.max_min_approx(alpha, U, problem, path_dict, 1000.0,
                                                                                 break_down=False)
    utils.write_to_file_as_pickle(swan_fid_to_flow_rate_mapping, LOG_DIR, swan_file_path)
# swan_fid_to_flow_rate_mapping = utils.read_pickle_file("./outputs/swan_gtsc_5.pickle")

print(file_name)
print("Approx w bet p mcf", approx_w_bet_p_mcf_dur)
# print("SWAN duration", swan_dur)
# print("danna duration", danna_dur)
total_approx_bet_p_mcf = 0
total_approx_bet = 0
total_swan = 0
total_danna = 0
fairness_swan = 0
fairness_approx_bet_p_mcf = 0
fairness_approx_bet = 0
fairness_swan_cdf = []
fairness_approx_bet_p_mcf_cdf = []
fairness_approx_bet_cdf = []
fairness_danna_cdf = []

approx_bet_rate_list = []
danna_rate_list = []
for fid, (src, dst, demand) in problem.sparse_commodity_list:
    baseline_rate = danna_fid_to_total_rate[src, dst]
    approx_bet_p_mcf_rate = np.sum(approx_water_bet_p_mcf_fid_to_flow_rate_mapping[src, dst])
    approx_bet_rate = np.sum(approx_water_bet_fid_to_flow_rate_mapping[src, dst])
    swan_rate = np.sum(swan_fid_to_flow_rate_mapping[src, dst])

    danna_rate_list.append(baseline_rate)
    approx_bet_rate_list.append(approx_bet_rate)

    total_approx_bet_p_mcf += approx_bet_p_mcf_rate
    total_swan += swan_rate
    total_danna += baseline_rate
    total_approx_bet += approx_bet_rate

    # print(approx_rate, baseline_rate, swan_rate)
    fairness_num = np.maximum(approx_bet_p_mcf_rate, 0.1) / np.maximum(baseline_rate, 0.1)
    fairness_approx_bet_p_mcf_cdf.append(fairness_num)
    # if fairness_num <= 0.3:
    #     print(src, dst, fairness_num, demand, approx_bet_p_mcf_rate, approx_bet_rate, baseline_rate, swan_rate)
    fairness_approx_bet_p_mcf += np.log10(np.minimum(fairness_num, 1.0 / fairness_num))

    fairness_num = np.maximum(swan_rate, 0.1) / np.maximum(baseline_rate, 0.1)
    fairness_swan_cdf.append(fairness_num)
    fairness_swan += np.log10(np.minimum(fairness_num, 1.0/fairness_num))

    fairness_num = np.maximum(approx_bet_rate, 0.1) / np.maximum(baseline_rate, 0.1)
    fairness_approx_bet_cdf.append(fairness_num)
    fairness_approx_bet += np.log10(np.minimum(fairness_num, 1.0/fairness_num))

    fairness_danna_cdf.append(1.0)

    # print(total_approx_1, total_approx_2)
    # assert np.abs(total_approx_1 - total_approx_2) <= constants.O_epsilon
fairness_swan = np.power(10, fairness_swan / len(problem.sparse_commodity_list))
fairness_approx_bet_p_mcf = np.power(10, fairness_approx_bet_p_mcf / len(problem.sparse_commodity_list))
fairness_approx_bet = np.power(10, fairness_approx_bet / len(problem.sparse_commodity_list))
print(total_danna, total_swan, total_approx_bet_p_mcf, total_approx_bet)
print(1.0, fairness_swan, fairness_approx_bet_p_mcf, fairness_approx_bet)
print(np.min(fairness_approx_bet_cdf), np.min(fairness_approx_bet_p_mcf_cdf))
plt.figure()
sns.ecdfplot(fairness_approx_bet_p_mcf_cdf, label="approx-bet-mcf")
sns.ecdfplot(fairness_swan_cdf, label="swan")
sns.ecdfplot(fairness_approx_bet_cdf, label="approx-bet")
sns.ecdfplot(fairness_danna_cdf, label="danna")
plt.xlim([0, 2])
plt.legend()
plt.show()

approx_bet_rate_list = np.array(approx_bet_rate_list)
danna_rate_list = np.array(danna_rate_list)
approx_bet_sorted_fids = np.argsort(approx_bet_rate_list)
danna_rate_list_fids = np.argsort(danna_rate_list)
num_errors = []
list_buckets = range(1, 100, 5)
for num_buckets in list_buckets:
    num_curr_err = 0
    # num_flows_per_bucket = len(problem.sparse_commodity_list) // num_buckets
    sliced_approx_bet_rate_list = np.array_split(approx_bet_sorted_fids, num_buckets)
    sliced_danna_rate_list = np.array_split(danna_rate_list_fids, num_buckets)
    for bucked_id, fid_list in enumerate(sliced_approx_bet_rate_list):
        for fid in fid_list:
            found = False
            for danna_bucket_id, fid_list in enumerate(sliced_danna_rate_list):
                if np.any(sliced_danna_rate_list[danna_bucket_id] == fid):
                    found = True
                    break
            assert found
            curr_bucket_min = np.min(danna_rate_list[sliced_danna_rate_list[bucked_id]])
            curr_bucket_max = np.max(danna_rate_list[sliced_danna_rate_list[bucked_id]])
            if bucked_id > danna_bucket_id and curr_bucket_min - danna_rate_list[fid] > 0.1:
                assert curr_bucket_min >= danna_rate_list[fid]
                num_curr_err += 1
            if bucked_id < danna_bucket_id and danna_rate_list[fid] - curr_bucket_max > 0.1:
                assert curr_bucket_max <= danna_rate_list[fid]
                num_curr_err += 1
    num_errors.append(num_curr_err * 100 / len(problem.sparse_commodity_list))
    print(num_buckets, num_errors[-1])
plt.figure()
plt.plot(list(list_buckets), num_errors, marker="X")
plt.xlabel("number of bins")
plt.ylabel("% error with impact")
plt.show()
raise Exception
approx_water_p_mcf_fid_to_flow_rate_mapping, approx_w_p_mcf = approx_water_plus_mcf.get_rates(problem, path_dict, path_edge_idx_mapping,
                                                                                              num_paths_per_flow, num_iter=1, link_cap=1000.0)



print("=" * 10 + " approx-waterfilling utilization " + "=" * 10)
utilization_approx_dict = defaultdict(int)
for (i, j), rate in approx_water_fid_to_flow_rate_mapping.items():
    for idx, path in enumerate(path_dict[i, j]):
        for link in zip(path, path[1:]):
            utilization_approx_dict[link] += rate[idx]

for link in utilization_approx_dict:
    # print(utilization_approx_dict[link], 1000)
    assert utilization_approx_dict[link] <= 1000 + 0.001


k_water_fid_to_flow_rate_mapping, water_dur = k_waterfilling.get_max_min_fair_multi_path_constrained_k_waterfilling(problem,
                                                                                                                    path_edge_idx_mapping,
                                                                                                                    num_paths_per_flow=num_paths_per_flow,
                                                                                                                    link_cap=1000.0,
                                                                                                                    k=1)
print("=" * 10 + " 1-waterfilling utilization " + "=" * 10)
utilization_waterfilling_dict = defaultdict(int)
for (i, j), rate in k_water_fid_to_flow_rate_mapping.items():
    for idx, path in enumerate(path_dict[i, j]):
        for link in zip(path, path[1:]):
            utilization_waterfilling_dict[link] += rate[idx]

for link in utilization_waterfilling_dict:
    # print(utilization_waterfilling_dict[link], 1000)
    assert utilization_waterfilling_dict[link] <= 1000 + 0.001



# swan_fid_to_flow_rate_mapping = swan_max_min_approx.compute_throughput_path_based_given_tm(problem, path_dict, 0, 1000, {}, 1000)
alpha = 2
U = 0.1
swan_fid_to_flow_rate_mapping, swan_dur = swan_max_min_approx.max_min_approx(alpha, U, problem, path_dict, 1000.0,
                                                                             break_down=True)

# total_flow_1 = 0
# total_flow_2 = 0
# for fid, (src, dst, demand) in problem.sparse_commodity_list:
#     total_approx_1 = np.sum(swan_fid_to_flow_rate_mapping_fst[src, dst])
#     total_approx_2 = np.sum(swan_fid_to_flow_rate_mapping[src, dst])
#     print(total_approx_1, total_approx_2, demand)
#     total_flow_1 += total_approx_1
#     total_flow_2 += total_approx_2
#     # assert np.abs(total_approx_1 - total_approx_2) <= constants.O_epsilon
# print("all flows are ok!")
# print("total flows:", total_flow_1, total_flow_2)
print("=" * 10 + " swan link utilization " + "=" * 10)
utilization_swan_dict = defaultdict(int)
for (i, j), rate in swan_fid_to_flow_rate_mapping.items():
    for idx, path in enumerate(path_dict[i, j]):
        for link in zip(path, path[1:]):
            utilization_swan_dict[link] += rate[idx]

for link in utilization_swan_dict:
    # print(link, utilization_swan_dict[link], 1000)
    assert utilization_swan_dict[link] <= 1000 + 0.001

total_flow_swan_to_danna = []
total_flow_kwater_to_danna = []
total_flow_approx_to_danna = []
total_flow_iewf_to_danna = []
total_flow_approx_w_mcf_to_danna = []
total_flow_approx_w_bet_to_danna = []
total_flow_approx_w_bet_mcf_to_danna = []
total_flow_danna_to_danna = []
sum_flow_danna = 0
sum_flow_swan = 0
sum_flow_kwater = 0
sum_flow_approx = 0
sum_flow_iewf = 0
sum_flow_approx_w_mcf = 0
sum_flow_approx_water_bet = 0
sum_flow_approx_water_bet_w_mcf = 0
for _, (src, dst, demand) in problem.sparse_commodity_list:
    total_flow_danna = danna_fid_to_total_rate[src, dst]
    total_flow_swan = sum(swan_fid_to_flow_rate_mapping[src, dst])
    total_flow_kwater = sum(k_water_fid_to_flow_rate_mapping[src, dst])
    total_flow_approx = sum(approx_water_fid_to_flow_rate_mapping[src, dst])
    # total_flow_iewf = sum(iewf_fid_to_flow_rate_mapping[src, dst])
    total_flow_approx_w_mcf = sum(approx_water_p_mcf_fid_to_flow_rate_mapping[src, dst])
    total_flow_approx_w_bet = sum(approx_water_bet_fid_to_flow_rate_mapping[src, dst])
    total_flow_approx_w_bet_mcf = sum(approx_water_bet_p_mcf_fid_to_flow_rate_mapping[src, dst])
    total_flow_approx_to_danna.append(np.log10(max(total_flow_approx, 0.1)/max(total_flow_danna, 0.1)))
    total_flow_kwater_to_danna.append(np.log10(max(total_flow_kwater, 0.1)/max(total_flow_danna, 0.1)))
    total_flow_swan_to_danna.append(np.log10(max(total_flow_swan, 0.1)/max(total_flow_danna, 0.1)))
    # total_flow_iewf_to_danna.append(np.log10(max(total_flow_iewf, 0.1)/max(total_flow_danna, 0.1)))
    total_flow_approx_w_mcf_to_danna.append(np.log10(max(total_flow_approx_w_mcf, 0.1)/max(total_flow_danna, 0.1)))
    total_flow_approx_w_bet_to_danna.append(np.log10(max(total_flow_approx_w_bet, 0.1)/max(total_flow_danna, 0.1)))
    total_flow_approx_w_bet_mcf_to_danna.append(np.log10(max(total_flow_approx_w_bet_mcf, 0.1)/max(total_flow_danna, 0.1)))
    total_flow_danna_to_danna.append(0)
    sum_flow_danna += total_flow_danna
    sum_flow_swan += total_flow_swan
    sum_flow_kwater += total_flow_kwater
    sum_flow_approx += total_flow_approx
    # sum_flow_iewf += total_flow_iewf
    sum_flow_approx_w_mcf += total_flow_approx_w_mcf
    sum_flow_approx_water_bet += total_flow_approx_w_bet
    sum_flow_approx_water_bet_w_mcf += total_flow_approx_w_bet_mcf
print("total flow danna", sum_flow_danna, "time", danna_dur)
print("total flow swan", sum_flow_swan, "time", swan_dur)
print("total flow kwater", sum_flow_kwater, "time", water_dur)
print("total flow approx", sum_flow_approx, "time", approx_dur)
# print("total flow iewf", sum_flow_iewf, "time", iewf_dur)
print("total flow approx + mcf", sum_flow_approx_w_mcf, "time", approx_w_p_mcf)
print("total flow approx w bet", sum_flow_approx_water_bet, "time", approx_w_bet_dur)
print("total flow approx w bet + mcf", sum_flow_approx_water_bet_w_mcf, "time", approx_w_bet_p_mcf_dur)
sns.ecdfplot(total_flow_danna_to_danna, label="danna")
sns.ecdfplot(total_flow_approx_to_danna, label="approx(1)")
sns.ecdfplot(total_flow_swan_to_danna, label="swan")
sns.ecdfplot(total_flow_kwater_to_danna, label="1-waterfilling")
# sns.ecdfplot(total_flow_iewf_to_danna, label="iewf(2)")
sns.ecdfplot(total_flow_approx_w_mcf_to_danna, label="approx(1) + MCF")
sns.ecdfplot(total_flow_approx_w_bet_to_danna, label="approx(1) bet")
sns.ecdfplot(total_flow_approx_w_bet_mcf_to_danna, label="approx(1) bet + MCF")
plt.legend()
plt.xlabel("Flow rate relative to Danna (log scale)")
plt.ylabel("Fraction of flows")
plt.grid()
plt.show()



sns.ecdfplot(total_flow_danna_to_danna, label="danna")
sns.ecdfplot(total_flow_approx_to_danna, label="approx(1)")
sns.ecdfplot(total_flow_swan_to_danna, label="swan")
sns.ecdfplot(total_flow_kwater_to_danna, label="1-waterfilling")
# sns.ecdfplot(total_flow_iewf_to_danna, label="iewf(2)")
sns.ecdfplot(total_flow_approx_w_mcf_to_danna, label="approx(1) + MCF")
sns.ecdfplot(total_flow_approx_w_bet_to_danna, label="approx(1) bet")
sns.ecdfplot(total_flow_approx_w_bet_mcf_to_danna, label="approx(1) bet + MCF")
plt.legend()
plt.xlim([-0.2, 0.2])
plt.xlabel("Flow rate relative to Danna (log scale)")
plt.ylabel("Fraction of flows")
plt.grid()
plt.show()

# portion_list_1water_swan = []
# potion_list_approx_swan = []
# swan_total_flow = 0
# waterfilling_total_flow = 0
# approx_total_flow = 0
# for (src, dst) in k_water_fid_to_flow_rate_mapping:
#     portion_list_1water_swan.append(np.log(sum(k_water_fid_to_flow_rate_mapping[src, dst]) / sum(swan_fid_to_flow_rate_mapping[src, dst])))
#     potion_list_approx_swan.append(np.log(sum(approx_water_fid_to_flow_rate_mapping[src, dst]) / sum(swan_fid_to_flow_rate_mapping[src, dst])))
#     waterfilling_total_flow += sum(k_water_fid_to_flow_rate_mapping[src, dst])
#     swan_total_flow += sum(swan_fid_to_flow_rate_mapping[src, dst])
#     approx_total_flow += sum(approx_water_fid_to_flow_rate_mapping[src, dst])
# sns.ecdfplot(portion_list_1water_swan, label="1water")
# sns.ecdfplot(potion_list_approx_swan, label="approx")
# plt.xlabel("rate of waterfilling relative to SWAN (log scale)")
# plt.legend()
# plt.show()
# print("Waterfilling total flow", waterfilling_total_flow)
# print("SWAN total flow", swan_total_flow)
# print("approx total flow", approx_total_flow)
#
# utilization_waterfilling = []
# utilization_swan = []
# utilization_approx = []
# link_set = problem.edges_list
# for link in link_set:
#     utilization_waterfilling.append(utilization_waterfilling_dict[link]/1000)
#     utilization_swan.append(utilization_swan_dict[link]/1000)
#     utilization_approx.append(utilization_approx_dict[link]/1000)
# sns.ecdfplot(utilization_swan, label="swan")
# sns.ecdfplot(utilization_waterfilling, label="waterfilling")
# sns.ecdfplot(utilization_approx, label="approx")
# plt.xlabel("link utilization")
# plt.legend()
# plt.show()
# print("done")
#
# print("1-water-runtime", water_dur)
# print("swan-runtime", swan_dur)
# print("approx-runtime", approx_dur)
