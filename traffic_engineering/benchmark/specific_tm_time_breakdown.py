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
num_iter_bet = 2


fnames = utils.find_topo_tm_fname(TOPO_NAME, TM_MODEL)
num_paths_per_flow = 16

# file_name = fnames[4]
found = False
for file_name in fnames:
    if "Cogentco.graphml_gravity_1229219359_64.0" in file_name[4]:
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

# approx_water_fid_to_flow_rate_mapping, approx_dur = approx_waterfilling.get_rates(problem, path_edge_idx_mapping,
#                                                                                   num_paths_per_flow, num_iter=1,
#                                                                                   link_cap=1000.0, break_down=True)
#
# k_water_fid_to_flow_rate_mapping, water_dur = k_waterfilling.get_max_min_fair_multi_path_constrained_k_waterfilling(problem,
#                                                                                                                     path_edge_idx_mapping,
#                                                                                                                     num_paths_per_flow=num_paths_per_flow,
#                                                                                                                     link_cap=1000.0,
#                                                                                                                     k=1,
#                                                                                                                     break_down=True)
#
# alpha = 2
# U = 0.1
# swan_fid_to_flow_rate_mapping, swan_dur = swan_max_min_approx.max_min_approx(alpha, U, problem, path_dict, 1000, break_down=True)
#
# print(file_name)
# print(approx_dur)
# print(water_dur)
# print(swan_dur)

APPROX_TOPO_NAME_TO_ITERATION = {
    "'Uninett2010.graphml'": (1, 10),
    "'Cogentco.graphml'": (1, 10),
    "'GtsCe.graphml'": (1, 10),
    "'UsCarrier.graphml'": (1, 20),
    "'Colt.graphml'": (1, 10),
    "'TataNld.graphml'": (1, 20),
    # 'Kdl.graphml',
}

danna_files = [("../outputs/danna_practical_2022_08_16_17_13_19_dc44cf7e", 0),
               ("../outputs/danna_practical_2022_08_16_22_22_21_6afe1d48", 0)]


approx_bet_files = [("../outputs/approx(1)_bet_2022_08_27_01_50_46_4284934e", 1)]

output_fairness_baseline = benchmark_plot_utils.read_rate_log_file(constants.DANNA, danna_files,
                                                                   "*", {constants.DANNA: "*"},
                                                                   APPROX_TOPO_NAME_TO_ITERATION)
output_approx_bet = benchmark_plot_utils.read_rate_log_file(constants.APPROX_BET, approx_bet_files,
                                                            "*", {constants.APPROX_BET: "*"},
                                                                   APPROX_TOPO_NAME_TO_ITERATION)


danna_fid_to_rate_mapping = output_fairness_baseline[2]
approx_bet_fid_to_rate_mapping = output_approx_bet[2]
utils.revise_list_commodities(problem)
danna_to_fid_rate_vector = np.zeros(shape=len(problem.sparse_commodity_list))
danna_fid_to_rate = utils.read_pickle_file(danna_fid_to_rate_mapping[num_paths_per_flow][f"'{file_name[4]}'"])
for fid, (src, dst, _) in problem.sparse_commodity_list:
    danna_to_fid_rate_vector[fid] = np.sum(danna_fid_to_rate[src, dst])
approx_fid_to_rate = utils.read_pickle_file(approx_bet_fid_to_rate_mapping[num_paths_per_flow][f"'{file_name[4]}'"])
approx_bet_fairness = \
    benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                 approx_fid_to_rate,
                                                                 problem.sparse_commodity_list,
                                                                 theta_fairness=0.1)


del output_fairness_baseline
del output_approx_bet

approx_output = approx_water_bet.get_rates(problem,
                                           path_split_ratio_mapping,
                                           num_paths_per_flow,
                                           num_iter_approx,
                                           num_iter_bet,
                                           link_cap,
                                           path_edge_idx_mapping=path_edge_idx_mapping,
                                           path_path_len_mapping=path_path_len_mapping)
fid_to_flow_rate_mapping, dur, _, run_time_dict = approx_output
fairness_no = \
        benchmark_plot_utils.compute_fairness_no_vectorized_baseline(
            danna_to_fid_rate_vector,
            fid_to_flow_rate_mapping,
            problem.sparse_commodity_list,
            theta_fairness=0.1)

total_flow = 0
for (src, dst) in fid_to_flow_rate_mapping:
    total_flow += np.sum(fid_to_flow_rate_mapping[src, dst])

print(fairness_no, approx_bet_fairness)
print(total_flow)


approx_output = approx_waterfilling.get_rates(problem, num_paths_per_flow, num_iter_approx,
                                              link_cap,
                                               path_edge_idx_mapping=path_edge_idx_mapping,
                                               path_path_len_mapping=path_path_len_mapping)

fid_to_flow_rate_mapping, dur, run_time_dict = approx_output
fairness_no = \
        benchmark_plot_utils.compute_fairness_no_vectorized_baseline(
            danna_to_fid_rate_vector,
            fid_to_flow_rate_mapping,
            problem.sparse_commodity_list,
            theta_fairness=0.1)

fairness_no_2 = benchmark_plot_utils.compute_fairness_no(danna_fid_to_rate,
                                                         fid_to_flow_rate_mapping, 0.1)

total_flow = 0
for (src, dst) in fid_to_flow_rate_mapping:
    total_flow += np.sum(fid_to_flow_rate_mapping[src, dst])

print(fairness_no, fairness_no_2, approx_bet_fairness)
print(total_flow)
