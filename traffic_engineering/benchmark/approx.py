import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from alg import approx_waterfilling, approx_water_plus_mcf, approx_water_bet, approx_water_bet_plus_mcf
from alg import waterfilling_utils
from utilities import shortest_paths, constants
from utilities import utils
from ncflow.lib.problem import Problem

def get_rates(approach, problem, path_output, num_paths_per_flow, num_approx_iter, num_bet_iter, link_cap,
              mcf_grb_method=2, num_bins=5, min_beta=1e-4, min_epsilon=1e-2, k=1, alpha=0, link_cap_scale_factor=1):
    path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output
    if approach == constants.APPROX:
        fid_to_flow_rate_mapping, dur, run_time_dict = approx_waterfilling.get_rates(problem, num_paths_per_flow,
                                                                                     num_iter_approx_water=num_approx_iter,
                                                                                     link_cap=link_cap,
                                                                                     path_edge_idx_mapping=path_edge_idx_mapping,
                                                                                     path_path_len_mapping=path_path_len_mapping)

    elif approach == constants.APPROX_MCF:
        fid_to_flow_rate_mapping, dur = approx_water_plus_mcf.get_rates(problem, path_dict, path_edge_idx_mapping,
                                                                        num_paths_per_flow, num_iter=num_approx_iter,
                                                                        link_cap=link_cap)
    elif approach == constants.APPROX_BET:
        fid_to_flow_rate_mapping, dur, _, run_time_dict = approx_water_bet.get_rates(problem,
                                                                                     path_split_ratio_mapping,
                                                                                     num_paths_per_flow,
                                                                                     num_iter_approx_water=num_approx_iter,
                                                                                     num_iter_bet=num_bet_iter,
                                                                                     link_cap=link_cap,
                                                                                     path_edge_idx_mapping=path_edge_idx_mapping,
                                                                                     path_path_len_mapping=path_path_len_mapping,
                                                                                     bias_toward_low_flow_rate=False)
    elif approach == constants.APPROX_BET_MCF:
        # problem: problem.Problem, paths, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
        # path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet, link_cap,
        # mcf_grb_method = 2, num_bins = 5, min_beta = 1e-4, min_epsilon = 1e-2, k = 1, alpha = 0, link_cap_scale_factor = 1,
        # break_down = False, biased_toward_low_flow_rate = False, biased_alpha = None
        min_eff_dur = np.inf
        min_dur = np.inf
        min_run_time_dict = None
        min_fid_to_flow_rate_mapping = None
        for i in range(num_repeats):
            fid_to_flow_rate_mapping, dur, run_time_dict = approx_water_bet_plus_mcf.get_rates(problem, path_dict, path_edge_idx_mapping,
                                                                                               path_path_len_mapping, path_total_path_len_mapping,
                                                                                               path_split_ratio_mapping, num_paths_per_flow,
                                                                                               num_iter_approx_water=num_approx_iter,
                                                                                               num_iter_bet=num_bet_iter,
                                                                                               link_cap=link_cap,
                                                                                               mcf_grb_method=mcf_grb_method,
                                                                                               num_bins=num_bins,
                                                                                               min_beta=min_beta,
                                                                                               min_epsilon=min_epsilon,
                                                                                               k=k,
                                                                                               alpha=alpha,
                                                                                               link_cap_scale_factor=link_cap_scale_factor)
            eff_dur = 0
            for name in import_dur:
                eff_dur += run_time_dict[name]

            if eff_dur < min_eff_dur:
                print(f"=== updating eff_dur from {min_eff_dur} to {eff_dur}")
                min_eff_dur = eff_dur
                min_dur = dur
                min_run_time_dict = run_time_dict
                min_fid_to_flow_rate_mapping = fid_to_flow_rate_mapping

        run_time_dict = min_run_time_dict
        dur = min_dur
        fid_to_flow_rate_mapping = min_fid_to_flow_rate_mapping

    elif approach == constants.APPROX_BET_MCF_BIASED:
        fid_to_flow_rate_mapping, dur = approx_water_bet_plus_mcf.get_rates(problem, path_dict, path_edge_idx_mapping,
                                                                            path_path_len_mapping, path_total_path_len_mapping,
                                                                            path_split_ratio_mapping, num_paths_per_flow,
                                                                            num_iter_approx_water=num_approx_iter,
                                                                            num_iter_bet=num_bet_iter, link_cap=link_cap,
                                                                            biased_toward_low_flow_rate=True)
    else:
        raise Exception(f"approach {approach} does not exist!")

    return fid_to_flow_rate_mapping, dur, run_time_dict


TM_MODEL_LIST = [
    'uniform',
    'bimodal',
    'gravity',
    'poisson-high-inter',
    # 'poisson-high-intra',
]
TOPO_NAME_LIST = [
    'Uninett2010.graphml',
    'Cogentco.graphml',
    'GtsCe.graphml',
    'UsCarrier.graphml',
    'Colt.graphml',
    'TataNld.graphml',
    # 'Kdl.graphml',
]
# APPROACH_LIST = [constants.APPROX_MCF, constants.APPROX_BET, constants.APPROX_BET_MCF]
# APPROACH_LIST = [constants.APPROX_BET, constants.APPROX_BET_MCF]
APPROACH_LIST = [
    # constants.APPROX,
    # constants.APPROX_BET,
    constants.APPROX_BET_MCF,
]
# num_path_list = [4, 16]
mcf_grb_method = 2
num_path_list = [16]
num_iter_approx_list = [1]
num_iter_bet_list = [10]
link_cap = 1000.0
log_dir = "../outputs"
base_split = 0.9
split_type = constants.EXPONENTIAL_DECAY
num_repeats = 3

# (num_bins, min_epsilon, min_beta, k, link_cap_scale_factor)
TOPOLOGY_TO_APPROX_BET_MCF_PARAMS = {
    'Uninett2010.graphml': (4, 1e-2, 1e-2, 1, 1000),
    'Cogentco.graphml': (15, 1e-2, 1e-2, 1, 1000),
    'GtsCe.graphml': (12, 1e-2, 1e-2, 1, 1000),
    'UsCarrier.graphml': (7, 1e-2, 1e-2, 1, 1000),
    'Colt.graphml': (10, 1e-2, 1e-2, 1, 1000),
    'TataNld.graphml': (15, 1e-2, 1e-2, 1, 1000),
    # 'Kdl.graphml',
}

import_dur = ["computation", "solver_time_equi_depth"]

for topo_name in TOPO_NAME_LIST:
    assert topo_name in TOPOLOGY_TO_APPROX_BET_MCF_PARAMS


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

                            per_flow_log_file_name = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths}"
                            per_flow_log_file_name += f"_iter_approx_{num_iter_approx}_iter_bet_{num_iter_bet}"
                            print("=" * 5, file_name, per_flow_log_file_name, "=" * 5)
                            num_bins, min_epsilon, min_beta, k, link_cap_scale_factor = TOPOLOGY_TO_APPROX_BET_MCF_PARAMS[topo_name]
                            approx_output = get_rates(approach, problem, path_output, num_paths,
                                                      num_iter_approx, num_iter_bet, link_cap,
                                                      mcf_grb_method=mcf_grb_method,
                                                      num_bins=num_bins,
                                                      min_beta=min_beta,
                                                      min_epsilon=min_epsilon,
                                                      k=k,
                                                      link_cap_scale_factor=link_cap_scale_factor)
                            fid_to_flow_rate_mapping, dur, run_time_dict = approx_output
                            total_flow = 0
                            for (src, dst) in fid_to_flow_rate_mapping:
                                total_flow += np.sum(fid_to_flow_rate_mapping[src, dst])

                            per_flow_log_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_per_path_flow.pickle"
                            utils.write_to_file_as_pickle(fid_to_flow_rate_mapping, log_folder_flows, per_flow_log_file_path)

                            run_time_dict_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_run_time_dict.pickle"
                            utils.write_to_file_as_pickle(run_time_dict, log_folder_flows, run_time_dict_file_path)

                            log_txt = f"{file_name},{num_paths},{num_iter_approx},{num_iter_bet},{total_flow}," \
                                      f"{dur}, {per_flow_log_file_name}\n"
                            utils.write_to_file(log_txt, log_dir, log_file)
                            print("results:", log_txt)
