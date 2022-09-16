import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from utilities import shortest_paths
from utilities import utils, constants
from ncflow.lib.problem import Problem
from alg import approx_water_bet_plus_mcf as equi_binner
from scripts import benchmark_plot_utils


TM_MODEL = 'gravity'
TOPO_NAME = 'Cogentco.graphml'
traffic_name_init = "Cogentco.graphml_gravity_1229219359_64.0_76765.140625"
split_type = constants.EXPONENTIAL_DECAY
bins_list = [1, 2, 4, 8, 16]


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

num_bins, min_epsilon, min_beta, k, link_cap_scale_factor, num_iter_approx, num_iter_bet, base_split = \
    constants.TOPOLOGY_TO_APPROX_BET_MCF_PARAMS[TOPO_NAME]
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
for fid, (src, dst, _) in problem.sparse_commodity_list:
    danna_to_fid_rate_vector[fid] = np.sum(danna_fid_to_rate[src, dst])

for num_bins in bins_list:
    print(f"num bins {num_bins}")
    eb_output = equi_binner.get_rates(problem, path_dict, path_edge_idx_mapping,
                                       path_path_len_mapping, path_total_path_len_mapping,
                                       path_split_ratio_mapping, num_paths_per_flow,
                                       num_iter_approx_water=num_iter_approx,
                                       num_iter_bet=num_iter_bet,
                                       link_cap=link_cap,
                                       mcf_grb_method=2,
                                       num_bins=num_bins,
                                       min_beta=min_beta,
                                       min_epsilon=min_epsilon,
                                       k=k,
                                       alpha=0,
                                       link_cap_scale_factor=link_cap_scale_factor)
    eb_fid_to_flow_rate_mapping, duration, run_time_dict = eb_output
    fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                               eb_fid_to_flow_rate_mapping,
                                                                               problem.sparse_commodity_list,
                                                                               theta_fairness=0.1)

    total_rate = 0
    for fid, (src, dst, demand) in problem.sparse_commodity_list:
        total_rate += np.sum(eb_fid_to_flow_rate_mapping[src, dst])
    print(fairness_no, total_rate, duration, run_time_dict)


