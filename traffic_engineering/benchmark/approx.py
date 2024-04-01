import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from alg import approx_waterfilling, approx_water_plus_mcf, approx_water_bet, approx_water_bet_plus_mcf
from alg import waterfilling_utils
from utilities import shortest_paths, constants
from utilities import utils
from ncflow.lib.problem import Problem


def get_rates(approach, problem, path_output, num_paths_per_flow, num_approx_iter, num_bet_iter, link_cap):
    path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output
    if approach == constants.APPROX:
        fid_to_flow_rate_mapping, dur = approx_waterfilling.get_rates(problem, path_edge_idx_mapping, num_paths_per_flow,
                                                                      num_iter=num_approx_iter, link_cap=link_cap)
    elif approach == constants.APPROX_MCF:
        fid_to_flow_rate_mapping, dur = approx_water_plus_mcf.get_rates(problem, path_dict, path_edge_idx_mapping,
                                                                        num_paths_per_flow, num_iter=num_approx_iter,
                                                                        link_cap=link_cap)
    elif approach == constants.APPROX_BET:
        fid_to_flow_rate_mapping, dur = approx_water_bet.get_rates(problem, path_edge_idx_mapping, path_path_len_mapping,
                                                                   path_total_path_len_mapping, path_split_ratio_mapping,
                                                                   num_paths_per_flow, num_iter_approx_water=num_approx_iter,
                                                                   num_iter_bet=num_bet_iter, link_cap=link_cap)
    elif approach == constants.APPROX_BET_MCF:
        # problem: problem.Problem, paths, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
        # path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet, link_cap, break_down = False,
        # biased_toward_low_flow_rate = False, biased_alpha = None):
        fid_to_flow_rate_mapping, dur = approx_water_bet_plus_mcf.get_rates(problem, path_dict, path_edge_idx_mapping,
                                                                            path_path_len_mapping, path_total_path_len_mapping,
                                                                            path_split_ratio_mapping, num_paths_per_flow,
                                                                            num_iter_approx_water=num_approx_iter,
                                                                            num_iter_bet=num_bet_iter, link_cap=link_cap)
    elif approach == constants.APPROX_BET_MCF_BIASED:
        fid_to_flow_rate_mapping, dur = approx_water_bet_plus_mcf.get_rates(problem, path_dict, path_edge_idx_mapping,
                                                                            path_path_len_mapping, path_total_path_len_mapping,
                                                                            path_split_ratio_mapping, num_paths_per_flow,
                                                                            num_iter_approx_water=num_approx_iter,
                                                                            num_iter_bet=num_bet_iter, link_cap=link_cap,
                                                                            biased_toward_low_flow_rate=True)
    else:
        raise Exception(f"approach {approach} does not exist!")

    return fid_to_flow_rate_mapping, dur


# TM_MODEL_LIST = ['bimodal', 'gravity']
TM_MODEL_LIST = ['bimodal', 'gravity']
TOPO_NAME_LIST = ['Cogentco.graphml', 'GtsCe.graphml']
# APPROACH_LIST = [constants.APPROX_MCF, constants.APPROX_BET, constants.APPROX_BET_MCF]
# APPROACH_LIST = [constants.APPROX_BET, constants.APPROX_BET_MCF]
APPROACH_LIST = [constants.APPROX_BET_MCF]
num_path_list = [16]
# num_iter_list = [1, 5, 10]
num_iter_approx_list = [1]
num_iter_bet_list = [10]
link_cap = 1000.0
log_dir = "../outputs"
base_split = 0.9
split_type = constants.EXPONENTIAL_DECAY

# current = 0
# skip_until = 70


for approach in APPROACH_LIST:
    fid = utils.get_fid()
    log_file = log_dir + "/{}_" + fid + ".txt"
    log_folder_flows = log_dir + "/{}_" + fid + "/"

    for num_paths in num_path_list:
        for topo_name in TOPO_NAME_LIST:
            for tm_model in TM_MODEL_LIST:
                fnames = utils.find_topo_tm_fname(topo_name, tm_model)
                example_file = fnames[0]
                problem = Problem.from_file(example_file[3], example_file[4])
                path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                                    k=num_paths, number_processors=32,
                                                                                    base_split=base_split,
                                                                                    split_type=split_type)
                for file_name in fnames:
                    problem = Problem.from_file(file_name[3], file_name[4])
                    utils.revise_list_commodities(problem)
                    for num_iter_approx in num_iter_approx_list:
                        for num_iter_bet in num_iter_bet_list:
                            log_file = log_file.format(waterfilling_utils.get_approx_label(approach, (num_iter_approx, num_iter_bet)))
                            log_folder_flows = log_folder_flows.format(waterfilling_utils.get_approx_label(approach, (num_iter_approx, num_iter_bet)))
                            # if current < skip_until:
                            #     current += 1
                            #     continue
                            per_flow_log_file_name = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths}"
                            per_flow_log_file_name += f"_iter_approx_{num_iter_approx}_iter_bet_{num_iter_bet}"
                            print("=" * 5, file_name, per_flow_log_file_name, "=" * 5)
                            fid_to_flow_rate_mapping, dur = get_rates(approach, problem, path_output, num_paths,
                                                                      num_iter_approx, num_iter_bet, link_cap)
                            total_flow = 0
                            for (src, dst) in fid_to_flow_rate_mapping:
                                total_flow += np.sum(fid_to_flow_rate_mapping[src, dst])

                            per_flow_log_file_path = log_folder_flows + "/" + per_flow_log_file_name + ".pickle"
                            utils.write_to_file_as_pickle(fid_to_flow_rate_mapping, log_folder_flows, per_flow_log_file_path)

                            log_txt = f"{file_name},{num_paths},{num_iter_approx},{num_iter_bet},{total_flow}," \
                                      f"{dur}, {per_flow_log_file_name}\n"
                            utils.write_to_file(log_txt, log_dir, log_file)
                            print("results:", log_txt)
