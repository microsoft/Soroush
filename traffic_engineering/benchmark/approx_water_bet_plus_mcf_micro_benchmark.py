import sys
import os
from collections import defaultdict

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from alg import approx_water_bet_plus_mcf
from alg import waterfilling_utils
from utilities import shortest_paths, constants
from utilities import utils
from ncflow.lib.problem import Problem
from scripts import benchmark_plot_utils


TM_MODEL_LIST = [
    'uniform',
    #'bimodal',
    #'gravity',
    #'poisson-high-inter',
    # 'poisson-high-intra',
]
TOPO_NAME_LIST = [
    #'Uninett2010.graphml',
    #'Cogentco.graphml',
    #'GtsCe.graphml',
    'UsCarrier.graphml',
    'Colt.graphml',
    'TataNld.graphml',
    # 'Kdl.graphml',
]

danna_files = [("../outputs/danna_practical_2022_08_16_17_13_19_dc44cf7e", 0),
               ("../outputs/danna_practical_2022_08_16_22_22_21_6afe1d48", 0)]

approx_bet_files = [("../outputs/approx(1)_bet_2022_08_27_01_50_46_4284934e", 1)]

output_fairness_baseline = benchmark_plot_utils.read_rate_log_file(constants.DANNA, danna_files,
                                                                   "*", {constants.DANNA: "*"})
output_approx_bet = benchmark_plot_utils.read_rate_log_file(constants.APPROX_BET, approx_bet_files,
                                                            "*", {constants.APPROX_BET: "*"})



danna_fid_to_rate_mapping = output_fairness_baseline[2]
approx_bet_fid_to_rate_mapping = output_approx_bet[2]
danna_to_fid_rate_vector_mapping = defaultdict(dict)
approx_bet_file_to_fairness_mapping = defaultdict(dict)
for tm_model in TM_MODEL_LIST:
    for topo_name in TOPO_NAME_LIST:
        fnames = utils.find_topo_tm_fname(topo_name, tm_model)
        for file_name in fnames:
            problem = Problem.from_file(file_name[3], file_name[4])
            utils.revise_list_commodities(problem)
            for num_path in danna_fid_to_rate_mapping:
                danna_to_fid_rate_vector_mapping[num_path][f"'{file_name[4]}'"] = \
                    np.zeros(shape=len(problem.sparse_commodity_list))
                danna_fid_to_rate = utils.read_pickle_file(danna_fid_to_rate_mapping[num_path][f"'{file_name[4]}'"])
                for fid, (src, dst, _) in problem.sparse_commodity_list:
                    danna_to_fid_rate_vector_mapping[num_path][f"'{file_name[4]}'"][fid] = \
                        np.sum(danna_fid_to_rate[src, dst])
                for num_iter in approx_bet_fid_to_rate_mapping[num_path][f"'{file_name[4]}'"]:
                    approx_fid_to_rate = utils.read_pickle_file(approx_bet_fid_to_rate_mapping[num_path][f"'{file_name[4]}'"][num_iter])
                    approx_bet_file_to_fairness_mapping[f"'{file_name[4]}'"][num_iter] = \
                        benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector_mapping[num_path][f"'{file_name[4]}'"],
                                                                                     approx_fid_to_rate,
                                                                                     problem.sparse_commodity_list,
                                                                                     theta_fairness=0.1)


del output_fairness_baseline
del output_approx_bet

approach = constants.APPROX_BET
num_path_list = [16]
num_iter_approx_list = [1]
num_iter_bet_list = [
    10,
    20,
    30,
]
grb_method_list = [
    # 1,
    2
]
min_beta_list = [
    # 0,
    1e-2,
    # 1e-4,
    # 1e-6,
    # 1e-8
]
min_epsilon_list = [
    1e-2,
    # 1e-4,
    # 1e-6,
    # 1e-8
]
# num_bin_list = [
#     # 2,
#     3,
#     4,
#     5,
#     7,
#     # 10,
#     # 12,
#     # 15,
#     # 20
#     "SWAN"
# ]
topology_to_num_bin_list = {
    'Uninett2010.graphml': [
        4,
        # 5,
        # 10,
        # 12,
        # 15,
    ],
    'Cogentco.graphml': [
        # 5,
        # 7,
        # "SWAN"
        # 10,
        # 12,
        15,
    ],
    'GtsCe.graphml': [
        # 5,
        # "SWAN"
        # 10,
        12,
        # 15,
    ],
    'UsCarrier.graphml': [
        7,
        # "SWAN"
        # 10,
        # 12,
        # 15,
    ],
    'Colt.graphml': [
        # 7,
        # "SWAN",
        10,
        # 12,
        # 15,
    ],
    'TataNld.graphml': [
        # 4,
        # 7,
        # "SWAN"
        # 10,
        # 12,
        15,
    ],
}
k_list = [
    1,
    # 10,
    # 100
]

cap_scale_factor_list = [
    # 1,
    # 100,
    1000
]
link_cap = 1000.0
log_dir = "../outputs"
fid = utils.get_fid()
log_file = f"../outputs/mini_benchmark_approx_w_bet_p_mcf_{fid}.txt"
log_folder_flows = f"../outputs/mini_benchmark_approx_w_bet_p_mcf_{fid}/"
base_split = 0.9
split_type = constants.EXPONENTIAL_DECAY
num_scenario_per_topo_traffic = 5
U = 0.1
alpha = 2
ignore_scale_factor_under = 16

for num_paths in num_path_list:
    for topo_name in TOPO_NAME_LIST:
        for tm_model in TM_MODEL_LIST:
            fnames = utils.find_topo_tm_fname(topo_name, tm_model)
            fnames_idx = np.random.choice(len(fnames), num_scenario_per_topo_traffic)
            example_file = fnames[0]
            problem = Problem.from_file(example_file[3], example_file[4])
            path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                                k=num_paths, number_processors=32,
                                                                                base_split=base_split,
                                                                                split_type=split_type)
            path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output
            num_bin_list = topology_to_num_bin_list[topo_name]

            for selected_fidx in fnames_idx:
                file_name = fnames[selected_fidx]
                tm_scale_factor = file_name[2]
                while tm_scale_factor < ignore_scale_factor_under:
                    new_fname_idx = np.random.choice(len(fnames), 1)[0]
                    if np.any(fnames_idx == new_fname_idx):
                        continue
                    file_name = fnames[new_fname_idx]
                    tm_scale_factor = file_name[2]

                for mcf_method in grb_method_list:
                    for scale_factor in cap_scale_factor_list:
                        for num_bins in num_bin_list:
                            for min_epsilon in min_epsilon_list:
                                for min_beta in min_beta_list:
                                    for k in k_list:
                                        problem = Problem.from_file(file_name[3], file_name[4])
                                        utils.revise_list_commodities(problem)
                                        if num_bins == "SWAN":
                                            max_demand = np.max(problem.traffic_matrix.tm)
                                            U = max(U, link_cap / len(problem.sparse_commodity_list))
                                            num_bins = 1 + int(np.ceil(np.log(max_demand / U) / np.log(alpha)))
                                        for num_iter_approx in num_iter_approx_list:
                                            for num_iter_bet in num_iter_bet_list:
                                                log_file = log_file.format(waterfilling_utils.get_approx_label(approach, (num_iter_approx, num_iter_bet)))
                                                log_folder_flows = log_folder_flows.format(waterfilling_utils.get_approx_label(approach, (num_iter_approx, num_iter_bet)))

                                                per_flow_log_file_name1 = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths}"
                                                per_flow_log_file_name = per_flow_log_file_name1 + \
                                                                         f"_iter_approx_{num_iter_approx}_iter_bet_{num_iter_bet}"
                                                print("=" * 5, file_name, per_flow_log_file_name, "=" * 5)

                                                approx_output = approx_water_bet_plus_mcf.get_rates(problem=problem,
                                                                                                    paths=path_dict,
                                                                                                    path_edge_idx_mapping=path_edge_idx_mapping,
                                                                                                    path_path_len_mapping=path_path_len_mapping,
                                                                                                    path_total_path_len_mapping=path_total_path_len_mapping,
                                                                                                    path_split_ratio_mapping=path_split_ratio_mapping,
                                                                                                    num_paths_per_flow=num_paths,
                                                                                                    num_iter_approx_water=num_iter_approx,
                                                                                                    num_iter_bet=num_iter_bet,
                                                                                                    link_cap=link_cap,
                                                                                                    mcf_grb_method=mcf_method,
                                                                                                    num_bins=num_bins,
                                                                                                    min_beta=min_beta,
                                                                                                    min_epsilon=min_epsilon,
                                                                                                    k=k,
                                                                                                    alpha=0,
                                                                                                    link_cap_scale_factor=scale_factor)
                                                fid_to_flow_rate_mapping, dur, run_time_dict = approx_output
                                                total_flow = 0
                                                for (src, dst) in fid_to_flow_rate_mapping:
                                                    total_flow += np.sum(fid_to_flow_rate_mapping[src, dst])

                                                fairness_no = \
                                                    benchmark_plot_utils.compute_fairness_no_vectorized_baseline(
                                                        danna_to_fid_rate_vector_mapping[num_paths][f"'{file_name[4]}'"],
                                                        fid_to_flow_rate_mapping,
                                                        problem.sparse_commodity_list,
                                                        theta_fairness=0.1)
                                                approx_bet_fairness_no = \
                                                    approx_bet_file_to_fairness_mapping[f"'{file_name[4]}'"][(str(num_iter_approx_list[0]), str(num_iter_bet_list[0]))]
                                                log_txt = f"{mcf_method},{scale_factor},{num_bins}," \
                                                          f"{min_epsilon},{min_beta},{k}," \
                                                          f"{file_name},{num_paths},{num_iter_approx}," \
                                                          f"{num_iter_bet},{total_flow},{dur}," \
                                                          f"{run_time_dict['solver_time_equi_depth']},{fairness_no}," \
                                                          f"{approx_bet_fairness_no}," \
                                                          f"{per_flow_log_file_name}\n"
                                                utils.write_to_file(log_txt, log_dir, log_file)
                                                print("results:", log_txt)
                                                for key, time_s in run_time_dict.items():
                                                    log_txt = f"    {key}: {time_s} s\n"
                                                    print("      ", log_txt)
