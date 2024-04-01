import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from alg import approx_waterfilling
from utilities import shortest_paths
from alg import swan_max_min_approx
from utilities import utils, constants
from ncflow.lib.problem import Problem
from alg import k_waterfilling, approx_water_bet, approx_waterfilling
from scripts import benchmark_plot_utils


TM_MODEL = 'gravity'
TOPO_NAME = 'Cogentco.graphml'
traffic_name_init = "Cogentco.graphml_gravity_1229219359_64.0_76765.140625"
base_split = .9
split_type = constants.EXPONENTIAL_DECAY
num_iter_approx = 1
num_iter_bet = 500

fnames = utils.find_topo_tm_fname(TOPO_NAME, TM_MODEL)
num_paths_per_flow = 16

# file_name = fnames[4]
found = False
for file_name in fnames:
    if traffic_name_init in file_name[4]:
        found = True
        break
assert found
problem = Problem.from_file(file_name[3], file_name[4])
link_cap = 1000.0

path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                    k=num_paths_per_flow, number_processors=32,
                                                                    base_split=base_split,
                                                                    split_type=split_type)
path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output

danna_files = [("../outputs/danna_practical_2022_08_16_17_13_19_dc44cf7e", 0),
               ("../outputs/danna_practical_2022_08_16_22_22_21_6afe1d48", 0)]

output_fairness_baseline = benchmark_plot_utils.read_rate_log_file(constants.DANNA, danna_files,
                                                                   "*", {constants.DANNA: "*"},
                                                                   {})

danna_fid_to_rate_mapping = output_fairness_baseline[2]
utils.revise_list_commodities(problem)
danna_to_fid_rate_vector = np.zeros(shape=len(problem.sparse_commodity_list))
danna_fid_to_rate = utils.read_pickle_file(danna_fid_to_rate_mapping[num_paths_per_flow][f"'{file_name[4]}'"])

print(f"=== num_comm: {len(problem.sparse_commodity_list)}")
for fid, (src, dst, _) in problem.sparse_commodity_list:
    danna_to_fid_rate_vector[fid] = np.sum(danna_fid_to_rate[src, dst])

approx_output = approx_water_bet.get_rates(problem,
                                           path_split_ratio_mapping,
                                           num_paths_per_flow,
                                           num_iter_approx,
                                           num_iter_bet,
                                           link_cap,
                                           path_edge_idx_mapping=path_edge_idx_mapping,
                                           path_path_len_mapping=path_path_len_mapping,
                                           return_details=True)
sample_allocation, sample_split_ratio = approx_output
iter_to_total_flow_mapping = dict()
iter_to_fairness_mapping = dict()
iter_to_split_ratio_change_mapping = dict()
for i in range(num_iter_bet):
    print(f"========== iter {i}")
    iter_to_total_flow_mapping[i] = np.sum(sample_allocation[i])
    print(f"= total flow {iter_to_total_flow_mapping[i]}")

    fid_to_rate_mapping = dict()
    for fid, (src, dst, _) in problem.sparse_commodity_list:
        fid_to_rate_mapping[(src, dst)] = np.sum(sample_allocation[i][fid, :])
    fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                               fid_to_rate_mapping,
                                                                               problem.sparse_commodity_list,
                                                                               theta_fairness=0.1)
    iter_to_fairness_mapping[i] = fairness_no
    print(f"= fairness no {iter_to_fairness_mapping[i]}")

    if i < num_iter_bet - 1:
        iter_to_split_ratio_change_mapping[i] = np.sum(np.abs(sample_split_ratio[i+1] - sample_split_ratio[i]))
        print(f"= split ratio change {iter_to_split_ratio_change_mapping[i]}")


# folder_name = f"../outputs/impact_iteration_approx_bet_{utils.get_fid()}"
# utils.write_to_file_as_pickle(iter_to_total_flow_mapping, folder_name, folder_name + "/total_flow.pickle")
# utils.write_to_file_as_pickle(iter_to_fairness_mapping, folder_name, folder_name + "/fairness.pickle")
# utils.write_to_file_as_pickle(iter_to_split_ratio_change_mapping, folder_name, folder_name + "/split_ratio_change.pickle")

