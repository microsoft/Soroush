import sys
import os
from collections import defaultdict

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from alg import approx_water_bet
from alg import waterfilling_utils
from utilities import shortest_paths, constants
from utilities import utils
from ncflow.lib.problem import Problem
from scripts import benchmark_plot_utils


TM_MODEL_LIST = [
    'uniform',
    'bimodal',
    'gravity',
    'poisson-high-inter',
    # 'poisson-high-intra',
]
TOPO_NAME_LIST = [
    'Uninett2010.graphml',
    #'Cogentco.graphml',
    #'GtsCe.graphml',
    # 'UsCarrier.graphml',
    # 'Colt.graphml',
    # 'TataNld.graphml',
    # 'Kdl.graphml',
]

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
                approx_fid_to_rate = utils.read_pickle_file(approx_bet_fid_to_rate_mapping[num_path][f"'{file_name[4]}'"])
                approx_bet_file_to_fairness_mapping[f"'{file_name[4]}'"] = \
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
    40,
]
link_cap = 1000.0
log_dir = "../outputs"
fid = utils.get_fid()
log_file = f"../outputs/mini_benchmark_approx_w_bet_{fid}.txt"
log_folder_flows = f"../outputs/mini_benchmark_approx_w_bet_{fid}/"
base_split_list = [
    0.7,
    0.9,
    0.95,
    1.0,
]
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
            change_mapping = dict()
            for base_split in base_split_list:
                example_file = fnames[0]
                problem = Problem.from_file(example_file[3], example_file[4])
                path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                                    k=num_paths, number_processors=32,
                                                                                    base_split=base_split,
                                                                                    split_type=split_type)
                path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output

                for selected_fidx in fnames_idx:
                    file_name = fnames[selected_fidx]
                    tm_scale_factor = file_name[2]
                    new_fname_idx = selected_fidx
                    while tm_scale_factor < ignore_scale_factor_under:
                        if selected_fidx in change_mapping:
                            new_fname_idx = change_mapping[selected_fidx]
                        else:
                            new_fname_idx = np.random.choice(len(fnames), 1)[0]
                        if np.any(fnames_idx == new_fname_idx):
                            continue
                        file_name = fnames[new_fname_idx]
                        tm_scale_factor = file_name[2]

                    change_mapping[selected_fidx] = new_fname_idx
                    problem = Problem.from_file(file_name[3], file_name[4])
                    utils.revise_list_commodities(problem)
                    for num_iter_approx in num_iter_approx_list:
                        for num_iter_bet in num_iter_bet_list:
                            log_file = log_file.format(waterfilling_utils.get_approx_label(approach, (num_iter_approx, num_iter_bet)))
                            log_folder_flows = log_folder_flows.format(waterfilling_utils.get_approx_label(approach, (num_iter_approx, num_iter_bet)))

                            per_flow_log_file_name1 = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths}"
                            per_flow_log_file_name = per_flow_log_file_name1 + \
                                                     f"_iter_approx_{num_iter_approx}_iter_bet_{num_iter_bet}"
                            print("=" * 5, file_name, per_flow_log_file_name, "=" * 5)

                            approx_output = approx_water_bet.get_rates(problem,
                                                                       path_split_ratio_mapping,
                                                                       num_paths,
                                                                       num_iter_approx,
                                                                       num_iter_bet,
                                                                       link_cap,
                                                                       path_edge_idx_mapping=path_edge_idx_mapping,
                                                                       path_path_len_mapping=path_path_len_mapping,
                                                                       )
                            fid_to_flow_rate_mapping, dur, _, run_time_dict = approx_output
                            total_flow = 0
                            for (src, dst) in fid_to_flow_rate_mapping:
                                total_flow += np.sum(fid_to_flow_rate_mapping[src, dst])

                            fairness_no = \
                                benchmark_plot_utils.compute_fairness_no_vectorized_baseline(
                                    danna_to_fid_rate_vector_mapping[num_paths][f"'{file_name[4]}'"],
                                    fid_to_flow_rate_mapping,
                                    problem.sparse_commodity_list,
                                    theta_fairness=0.1)
                            approx_bet_fairness_no = approx_bet_file_to_fairness_mapping[f"'{file_name[4]}'"]
                            log_txt = f"{base_split},{file_name},{num_paths},{num_iter_approx}," \
                                      f"{num_iter_bet},{total_flow},{dur}," \
                                      f"{run_time_dict['computation']},{fairness_no}," \
                                      f"{approx_bet_fairness_no}," \
                                      f"{per_flow_log_file_name}\n"
                            utils.write_to_file(log_txt, log_dir, log_file)
                            print("results:", log_txt)
                            for key, time_s in run_time_dict.items():
                                log_txt = f"    {key}: {time_s} s\n"
                                print("      ", log_txt)

