import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from utilities import shortest_paths
from utilities import utils, constants
from ncflow.lib.problem import Problem
from alg import geometric_approx_binning
from scripts import benchmark_plot_utils


TM_MODEL = 'poisson-high-inter'
TOPO_NAME = 'Cogentco.graphml'
SCALE_FACTOR = 64
# traffic_name_init = "Cogentco.graphml_gravity_1229219359_64.0_76765.140625"
base_split = .9
split_type = constants.EXPONENTIAL_DECAY
num_iter_approx = 1
num_iter_bet = 500
U = 0.1
min_epsilon = 1e-6
bins_list = [1, 2, 4, 8, 16, 32, 64]


# fnames = utils.find_topo_tm_fname(TOPO_NAME, TM_MODEL)
fnames = utils.find_topo_tm_scale_factor_fname(TOPO_NAME, TM_MODEL, SCALE_FACTOR)
num_paths_per_flow = 16

# # file_name = fnames[4]
# found = False
# for file_name in fnames:
#     if traffic_name_init in file_name[4]:
#         found = True
#         break
# assert found

problem = Problem.from_file(fnames[0][3], fnames[0][4])
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
danna_fid_to_total_flow_mapping = output_fairness_baseline[0]

for file_name in fnames:
    problem = Problem.from_file(file_name[3], file_name[4])
    link_cap = 1000.0

    utils.revise_list_commodities(problem)
    danna_to_fid_rate_vector = np.zeros(shape=len(problem.sparse_commodity_list))
    danna_fid_to_rate = utils.read_pickle_file(danna_fid_to_rate_mapping[num_paths_per_flow][f"'{file_name[4]}'"])
    for fid, (src, dst, _) in problem.sparse_commodity_list:
        danna_to_fid_rate_vector[fid] = np.sum(danna_fid_to_rate[src, dst])

    for num_bins in bins_list:
        max_demand = np.max(problem.traffic_matrix.tm)
        U = max(U, link_cap / len(problem.sparse_commodity_list))
        alpha = np.power(max_demand / U, 1.0 / num_bins)
        # print(f"====== alpha {alpha} max demand {max_demand} num bins {num_bins}")
        gb_output = geometric_approx_binning.max_min_approx(alpha, U, problem, path_dict, link_cap, min_epsilon)
        gb_fid_to_flow_rate_mapping, duration, run_time_dict = gb_output
        fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                gb_fid_to_flow_rate_mapping,
                                                                                problem.sparse_commodity_list,
                                                                                theta_fairness=0.1)

        effective_run_time = benchmark_plot_utils.get_output_run_time(constants.NEW_APPROX,
                                                                      run_time_dict,
                                                                      constants.approach_to_valid_for_run_time)
        total_rate = 0
        for fid, (src, dst, demand) in problem.sparse_commodity_list:
            total_rate += np.sum(gb_fid_to_flow_rate_mapping[src, dst])
        
        danna_efficiency = danna_fid_to_total_flow_mapping[num_paths_per_flow][f"'{file_name[4]}'"]

        # print(fairness_no, total_rate, duration, run_time_dict)
        print(f"=========== file name {file_name[4]} ============== \n" +
              f"alpha = {alpha}, \n" +
              f"num bins = {num_bins}, \n" +
              f"max demand = {max_demand}, \n" + 
              f"fairness = {fairness_no}, \n" +
              f"total flow = {total_rate}, \n" + 
              f"total run time = {duration}, \n" +
              f"effective run time = {effective_run_time} \n" 
              f"run time dict = {run_time_dict}\n")
        print(f"danna efficiency = {danna_efficiency}\n")
