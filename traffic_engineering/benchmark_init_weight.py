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
from alg import approx_water_plus_mcf, approx_water_bet, k_water_bet
from alg import approx_water_bet_plus_mcf
from alg import new_approx_waterfilling

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
num_paths_per_flow = 16

num_iter_bet = 10
file_name = fnames[f_idx]
problem = Problem.from_file(file_name[3], file_name[4])
utils.revise_list_commodities(problem)
print(file_name)

output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx, k=num_paths_per_flow, number_processors=32,
                                                               base_split=.9, split_type=constants.EXPONENTIAL_DECAY)
path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, len_biased_path_split_ratio_mapping = output

danna_file_path = f"./outputs/examples/{TOPO_NAME}/{TM_MODEL}/danna_gtsc_{f_idx}.pickle"
if utils.file_exists(danna_file_path):
    try:
        danna_fid_to_total_rate, danna_flow_id_to_flow_rate_mapping, danna_dur = utils.read_pickle_file(danna_file_path)
    except:
        danna_fid_to_total_rate = utils.read_pickle_file(danna_file_path)
else:
    danna_fid_to_total_rate, danna_flow_id_to_flow_rate_mapping, danna_dur = \
        danna_practical_max_min_fair.get_rates(0.1, problem, path_dict, link_cap=1000.0)
    utils.write_to_file_as_pickle((danna_fid_to_total_rate, danna_flow_id_to_flow_rate_mapping, danna_dur), LOG_DIR, danna_file_path)

# print(danna_fid_to_total_rate)
seed = 1
# print(f"========= seed {seed}")
rng = np.random.RandomState(seed=seed)
coeff = np.power(0.9, np.arange(num_paths_per_flow))
sorted_split_ratio = coeff / np.sum(coeff)
num_flows = len(problem.sparse_commodity_list)
random_path_split_ratio_mapping = dict()
for (src, dst) in len_biased_path_split_ratio_mapping:
    num_paths = np.count_nonzero(len_biased_path_split_ratio_mapping[src, dst])
    weight = rng.permutation(coeff)
    # weight = rng.random(size=16)
    random_path_split_ratio_mapping[src, dst] = np.zeros(num_paths_per_flow)
    random_path_split_ratio_mapping[src, dst][:num_paths] = weight[:num_paths] / np.sum(weight[:num_paths])

edge_to_weight_mapping = defaultdict(int)
for _, (src, dst, demand) in problem.sparse_commodity_list:
    for pid in path_edge_idx_mapping[src, dst]:
        for idx in path_edge_idx_mapping[src, dst][pid]:
            # edge_to_weight_mapping[idx] += 1
            edge_to_weight_mapping[idx] += np.power(0.9, len(path_edge_idx_mapping[src, dst][pid]))

path_split_ratio_mapping = dict()
for _, (src, dst, _) in problem.sparse_commodity_list:
    path_split_ratio_mapping[src, dst] = np.zeros(num_paths_per_flow)
    for pid in path_edge_idx_mapping[src, dst]:
        weight = 0
        for eid in path_edge_idx_mapping[src, dst][pid]:
            # weight += edge_to_weight_mapping[eid]
            weight = max(weight, edge_to_weight_mapping[eid])
        if weight > 0:
            path_split_ratio_mapping[src, dst][pid] = 1.0 / weight
    path_split_ratio_mapping[src, dst] = path_split_ratio_mapping[src, dst] / np.sum(path_split_ratio_mapping[src, dst])
    path_split_ratio_mapping[src, dst] = path_split_ratio_mapping[src, dst] + np.random.normal(loc=0, scale=0.1, size=num_paths_per_flow)
    path_split_ratio_mapping[src, dst] = np.maximum(0, path_split_ratio_mapping[src, dst])
    path_split_ratio_mapping[src, dst] /= np.sum(path_split_ratio_mapping[src, dst])

# danna_path_split_ratio_mapping = dict()
# for (src, dst) in danna_flow_id_to_flow_rate_mapping:
#     weights = danna_flow_id_to_flow_rate_mapping[src, dst] / np.sum(danna_flow_id_to_flow_rate_mapping[src, dst])
#     danna_path_split_ratio_mapping[src, dst] = np.zeros(num_paths_per_flow)
#     danna_path_split_ratio_mapping[src, dst][:len(danna_flow_id_to_flow_rate_mapping[src, dst])] = weights
#     assert np.abs(np.sum(danna_flow_id_to_flow_rate_mapping[src, dst]) - danna_fid_to_total_rate[src, dst]) <= 0.001
#     # if (np.random.random() <= 0.5):
#     #     path_split_ratio_mapping[src, dst] = danna_path_split_ratio_mapping[src, dst]
#     path_split_ratio_mapping[src, dst] = danna_path_split_ratio_mapping[src, dst] + np.random.normal(loc=0, scale=0.5, size=num_paths_per_flow)
#     path_split_ratio_mapping[src, dst] = np.maximum(0, path_split_ratio_mapping[src, dst])
#     path_split_ratio_mapping[src, dst] /= np.sum(path_split_ratio_mapping[src, dst])

print(path_split_ratio_mapping[0, 1])
print(len_biased_path_split_ratio_mapping[0, 1])
print(random_path_split_ratio_mapping[0, 1])
approx_water_bet_fid_to_flow_rate_mapping, approx_w_bet_dur, _ = k_water_bet.get_rates(problem,
                                                                                       path_split_ratio_mapping,
                                                                                       num_paths_per_flow,
                                                                                       num_iter_approx_water=1,
                                                                                       num_iter_bet=num_iter_bet,
                                                                                       link_cap=1000.0,
                                                                                       path_edge_idx_mapping=path_edge_idx_mapping,
                                                                                       path_path_len_mapping=path_path_len_mapping,
                                                                                       bias_toward_low_flow_rate=False,
                                                                                       bias_alpha=0.5)

alpha = 2
U = 0.1

approx_bet_path_split_ratio_mapping = dict()
for (src, dst) in approx_water_bet_fid_to_flow_rate_mapping:
    weights = approx_water_bet_fid_to_flow_rate_mapping[src, dst] / np.sum(approx_water_bet_fid_to_flow_rate_mapping[src, dst])
    approx_bet_path_split_ratio_mapping[src, dst] = np.zeros(num_paths_per_flow)
    approx_bet_path_split_ratio_mapping[src, dst][:len(approx_water_bet_fid_to_flow_rate_mapping[src, dst])] = weights

swan_file_path = f"./outputs/examples/{TOPO_NAME}/{TM_MODEL}/swan_gtsc_{f_idx}.pickle"
if utils.file_exists(swan_file_path):
    swan_fid_to_flow_rate_mapping = utils.read_pickle_file(swan_file_path)
else:
    swan_fid_to_flow_rate_mapping, swan_dur = swan_max_min_approx.max_min_approx(alpha, U, problem, path_dict, 1000.0,
                                                                                 break_down=False)
    utils.write_to_file_as_pickle(swan_fid_to_flow_rate_mapping, LOG_DIR, swan_file_path)
# swan_fid_to_flow_rate_mapping = utils.read_pickle_file("./outputs/swan_gtsc_5.pickle")

print(file_name)
# print("SWAN duration", swan_dur)
# print("danna duration", danna_dur)
total_approx_bet = 0
total_swan = 0
total_danna = 0
fairness_swan = 0
fairness_approx_bet = 0
fairness_swan_cdf = []
fairness_approx_bet_cdf = []
fairness_danna_cdf = []

approx_bet_rate_list = []
danna_rate_list = []
for fid, (src, dst, demand) in problem.sparse_commodity_list:
    baseline_rate = danna_fid_to_total_rate[src, dst]
    approx_bet_rate = np.sum(approx_water_bet_fid_to_flow_rate_mapping[src, dst])
    swan_rate = np.sum(swan_fid_to_flow_rate_mapping[src, dst])

    danna_rate_list.append(baseline_rate)
    approx_bet_rate_list.append(approx_bet_rate)

    total_swan += swan_rate
    total_danna += baseline_rate
    total_approx_bet += approx_bet_rate

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
fairness_approx_bet = np.power(10, fairness_approx_bet / len(problem.sparse_commodity_list))
print(total_danna, total_swan, total_approx_bet)
print(1.0, fairness_swan, fairness_approx_bet)
print(np.min(fairness_approx_bet_cdf))
# plt.figure()
# sns.ecdfplot(fairness_swan_cdf, label="swan")
# sns.ecdfplot(fairness_approx_bet_cdf, label="approx-bet")
# sns.ecdfplot(fairness_danna_cdf, label="danna")
# plt.xlim([0, 2])
# plt.legend()
# plt.show()