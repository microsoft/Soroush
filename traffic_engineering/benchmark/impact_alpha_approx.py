import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from utilities import shortest_paths
from utilities import utils, constants
from ncflow.lib.problem import Problem
from alg import geometric_approx_binning, swan_max_min_approx
from scripts import benchmark_plot_utils


TM_MODEL = 'gravity'
TOPO_NAME = 'Cogentco.graphml'
traffic_name_init = "Cogentco.graphml_gravity_1229219359_64.0_76765.140625"
base_split = .9
split_type = constants.EXPONENTIAL_DECAY
num_iter_approx = 1
num_iter_bet = 500
U = 0.1
# min_epsilon = 1e-6
alpha_min_epsilon_list = [
    # (2.0, 1e-6),
    (1.05, 1e-12),
    # (1.1, 1e-8),
    # (1.5, 1e-6),
    # (2.5, 1e-6),
    # (3.0, 1e-6),
    # (4.0, 1e-6),
    # (6.0, 1e-6),
    # (10.0, 1e-6)
]

output_dir = f"./outputs/impact_alpha_{utils.get_fid()}/"


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
for fid, (src, dst, _) in problem.sparse_commodity_list:
    danna_to_fid_rate_vector[fid] = np.sum(danna_fid_to_rate[src, dst])


for alpha, min_epsilon in alpha_min_epsilon_list:
    max_demand = np.max(problem.traffic_matrix.tm)
    U = max(U, link_cap / len(problem.sparse_commodity_list))
    num_bins = np.ceil(np.log(max_demand / U) / np.log(alpha))
    print(f"====== alpha {alpha} max demand {max_demand} num bins {num_bins}")
    gb_output = geometric_approx_binning.max_min_approx(alpha, U, problem, path_dict, link_cap, min_epsilon)
    gb_fid_to_flow_rate_mapping, gb_duration, gb_run_time_dict = gb_output
    gb_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                  gb_fid_to_flow_rate_mapping,
                                                                                  problem.sparse_commodity_list,
                                                                                  theta_fairness=0.1)

    # swan_output = swan_max_min_approx.max_min_approx(alpha, U, problem, path_dict, link_cap)
    # swan_fid_to_flow_rate_mapping, swan_dur, swan_run_time_dict = swan_output

    # swan_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
    #                                                                                 swan_fid_to_flow_rate_mapping,
    #                                                                                 problem.sparse_commodity_list,
    #                                                                                 theta_fairness=0.1)

    # total_rate_swan = 0
    total_rate_gb = 0
    for fid, (src, dst, demand) in problem.sparse_commodity_list:
        # total_rate_swan += np.sum(swan_fid_to_flow_rate_mapping[src, dst])
        total_rate_gb += np.sum(gb_fid_to_flow_rate_mapping[src, dst])

    log_txt = f"===================== results for alpha {alpha} & num bins {num_bins} ====================\n" \
            #   f"========= approach {constants.SWAN}\n" \
            #   f"fairness {swan_fairness_no}\n" \
            #   f"efficiency {total_rate_swan}\n" \
            #   f"run time {swan_run_time_dict}\n" \
              f"======== approach {constants.NEW_APPROX}\n" \
              f"fairness {gb_fairness_no}\n" \
              f"efficiecny {total_rate_gb}\n" \
              f"run time {gb_run_time_dict}\n"

    output_file = output_dir + f"alpha_approx_{alpha}.txt"
    utils.write_to_file(log_txt, output_dir, output_file)
    print(log_txt)
