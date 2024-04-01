import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from utilities import shortest_paths
from utilities import utils, constants
from ncflow.lib.problem import Problem
from alg import approx_water_bet_plus_mcf as equi_binner
from alg import geometric_approx_binning as gb_solver
from alg import danna_practical_max_min_fair as danna_solver
from alg import swan_max_min_approx as swan_solver
from alg import approx_water_bet as adaptive_water
from scripts import benchmark_plot_utils


# TM_MODEL = 'gravity'
TM_MODEL = 'poisson-high-inter'
TOPO_NAME = 'Cogentco.graphml'
SCALE_FACTOR = 64
# traffic_name_init = "Cogentco.graphml_gravity_1229219359_64.0_76765.140625"
split_type = constants.EXPONENTIAL_DECAY
num_path_list = [
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32
]
num_scenario_per_topo_traffic = 1

# fnames = utils.find_topo_tm_fname(TOPO_NAME, TM_MODEL)
fnames = utils.find_topo_tm_scale_factor_fname(TOPO_NAME, TM_MODEL, SCALE_FACTOR)

rng_problem_chooser = np.random.RandomState(seed=0)
fnames_idx = rng_problem_chooser.choice(len(fnames), num_scenario_per_topo_traffic)

# file_name = fnames[4]
# found = False
# for file_name in fnames:
#     if traffic_name_init in file_name[4]:
#         found = True
#         break
# assert found


link_cap = 1000.0
output_dir = f"./outputs/impact_path_{utils.get_fid()}/topo_{TOPO_NAME}/tm_{TM_MODEL}/sf_{SCALE_FACTOR}/"
utils.ensure_dir(output_dir)



num_bins, min_epsilon, min_beta, k, link_cap_scale_factor, num_iter_approx, num_iter_bet, base_split = \
    constants.TOPOLOGY_TO_APPROX_BET_MCF_PARAMS[TOPO_NAME]

for num_paths_per_flow in num_path_list:
    
    problem = Problem.from_file(fnames[0][3], fnames[0][4])
    utils.revise_list_commodities(problem)
    path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                        k=num_paths_per_flow, number_processors=32,
                                                                        base_split=base_split,
                                                                        split_type=split_type)
    path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output

    for selected_fidx in fnames_idx:
        file_name = fnames[selected_fidx]
        problem = Problem.from_file(file_name[3], file_name[4])
        utils.revise_list_commodities(problem)

        danna_output = danna_solver.get_rates(0, problem, path_dict, link_cap)
        danna_fid_to_total_rate, danna_fid_to_per_path_rate, danna_dur, danna_run_time_dict = danna_output

        danna_to_fid_rate_vector = np.zeros(shape=len(problem.sparse_commodity_list))
        for fid, (src, dst, _) in problem.sparse_commodity_list:
            danna_to_fid_rate_vector[fid] = np.sum(danna_fid_to_per_path_rate[src, dst])

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
        eb_fid_to_flow_rate_mapping, eb_duration, eb_run_time_dict = eb_output

        eb_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                    eb_fid_to_flow_rate_mapping,
                                                                                    problem.sparse_commodity_list,
                                                                                    theta_fairness=0.1)

        adaptive_output = adaptive_water.get_rates(problem, path_split_ratio_mapping, num_paths_per_flow,
                                                num_iter_approx, num_iter_bet, link_cap,
                                                path_edge_idx_mapping=path_edge_idx_mapping,
                                                path_path_len_mapping=path_path_len_mapping,
                                                bias_toward_low_flow_rate=False)
        adaptive_fid_to_flow_rate_mapping, adaptive_dur, _, adaptive_run_time_dict = adaptive_output

        adaptive_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                            adaptive_fid_to_flow_rate_mapping,
                                                                                            problem.sparse_commodity_list,
                                                                                            theta_fairness=0.1)

        swan_output = swan_solver.max_min_approx(2, 0.1, problem, path_dict, link_cap)
        swan_fid_to_flow_rate_mapping, swan_dur, swan_run_time_dict = swan_output

        swan_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                        swan_fid_to_flow_rate_mapping,
                                                                                        problem.sparse_commodity_list,
                                                                                        theta_fairness=0.1)

        gb_output = gb_solver.max_min_approx(2, 0.1, problem, path_dict, link_cap, max_epsilon=1e-6)
        gb_fid_to_flow_rate_mapping, gb_duration, gb_run_time_dict = gb_output

        gb_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                    gb_fid_to_flow_rate_mapping,
                                                                                    problem.sparse_commodity_list,
                                                                                    theta_fairness=0.1)

        total_rate_eb = 0
        total_rate_danna = 0
        total_rate_h = 0
        total_rate_swan = 0
        total_rate_gb = 0
        for fid, (src, dst, demand) in problem.sparse_commodity_list:
            total_rate_eb += np.sum(eb_fid_to_flow_rate_mapping[src, dst])
            total_rate_swan += np.sum(swan_fid_to_flow_rate_mapping[src, dst])
            total_rate_h += np.sum(adaptive_fid_to_flow_rate_mapping[src, dst])
            total_rate_danna += np.sum(danna_fid_to_total_rate[src, dst])
            total_rate_gb += np.sum(gb_fid_to_flow_rate_mapping[src, dst])

        eb_effective_run_time = benchmark_plot_utils.get_output_run_time(constants.APPROX_BET_MCF,
                                                                         eb_run_time_dict,
                                                                         constants.approach_to_valid_for_run_time)
        
        gb_effective_run_time = benchmark_plot_utils.get_output_run_time(constants.NEW_APPROX,
                                                                         gb_run_time_dict,
                                                                         constants.approach_to_valid_for_run_time)
        
        swan_effective_run_time = benchmark_plot_utils.get_output_run_time(constants.SWAN,
                                                                           swan_run_time_dict,
                                                                           constants.approach_to_valid_for_run_time)
        
        danna_effective_run_time = benchmark_plot_utils.get_output_run_time(constants.DANNA,
                                                                            danna_run_time_dict,
                                                                            constants.approach_to_valid_for_run_time)
        
        adaptive_effective_run_time = benchmark_plot_utils.get_output_run_time(constants.APPROX_BET,
                                                                               adaptive_run_time_dict,
                                                                               constants.approach_to_valid_for_run_time)

        log_txt = f"===================== file name {file_name[4]} ====================\n" \
                f"===================== results for num paths {num_paths_per_flow} ====================\n" \
                f"========= approach {constants.SWAN}\n" \
                f"fairness {swan_fairness_no}\n" \
                f"efficiency {total_rate_swan}\n" \
                f"effective run time {swan_effective_run_time}\n" \
                f"run time {swan_run_time_dict}\n" \
                f"========= approach {constants.APPROX_BET}\n" \
                f"fairness {adaptive_fairness_no}\n" \
                f"efficiency {total_rate_h}\n" \
                f"effective run time {adaptive_effective_run_time}\n" \
                f"run time {adaptive_run_time_dict}\n" \
                f"======== approach {constants.APPROX_BET_MCF}\n" \
                f"fairness {eb_fairness_no}\n" \
                f"efficiency {total_rate_eb}\n" \
                f"effective run time {eb_effective_run_time}\n" \
                f"run time {eb_run_time_dict}\n" \
                f"======== approach {constants.DANNA}\n" \
                f"fairness 1.0\n" \
                f"efficiency {total_rate_danna}\n" \
                f"effective run time {danna_effective_run_time}\n" \
                f"run time {danna_run_time_dict}\n" \
                f"======== approach {constants.NEW_APPROX}\n" \
                f"fairness {gb_fairness_no}\n" \
                f"efficiecny {total_rate_gb}\n" \
                f"effective run time {gb_effective_run_time}\n" \
                f"run time {gb_run_time_dict}\n"

        output_file = output_dir + f"shortest_paths_{num_paths_per_flow}.txt"
        utils.write_to_file(log_txt, output_dir, output_file)
