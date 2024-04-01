import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))

from utilities import shortest_paths, utils
from alg import swan_max_min_approx, pop_swan
from utilities import constants
from ncflow.lib.problem import Problem
from alg import danna_practical_max_min_fair
from alg import approx_water_bet_plus_mcf as equi_binner
# from alg import popv1_approx_water_bet_plus_mcf as popv1_equi_binner
from alg import popv2_approx_water_bet_plus_mcf as popv2_equi_binner
from alg import geometric_approx_binning, pop_geometric_approx_binning
from scripts import benchmark_plot_utils

import danna_practical


# TM_MODEL = 'gravity'
# TOPO_NAME = 'Cogentco.graphml'
# traffic_name_init_list = [
#     "Cogentco.graphml_gravity_1374632450_1.0_1201.365600",
#     "Cogentco.graphml_gravity_1176656848_2.0_2402.7595",
#     "Cogentco.graphml_gravity_1425924102_4.0_4809.27294",
#     "Cogentco.graphml_gravity_1877587945_8.0_9602.7597",
#     "Cogentco.graphml_gravity_1085483962_16.0_19241.53125",
#     "Cogentco.graphml_gravity_1790879274_32.0_38281.6406",
#     "Cogentco.graphml_gravity_1229219359_64.0_76765.140625",
#     "Cogentco.graphml_gravity_1882736072_128.0_153591.515",
# ]
TM_NUMERIC_MODEL_LIST = [
    # ('uniform', -1),
    # ('bimodal', -1),
    # ('gravity', -1),
    # ('poisson-high-inter', -1),
    ('poisson-high-intra', 5),
]
TOPO_NAME_LIST = [
    'Cogentco.graphml',
    'GtsCe.graphml',
    # 'UsCarrier.graphml',
    # 'TataNld.graphml',
]
SCALE_FACTOR_LIST = [
    # 1,
    # 2,
    # 4,
    # 8,
    16,
    # 32,
    64,
    # 128,
]

# general parameters
mcf_grb_method = 2
num_scenario_per_topo_traffic = 3
num_sub_problems_list = [
    2,
    4,
    8,
    16,
    48,
]
split_type = constants.EXPONENTIAL_DECAY
num_threads = 48

# log
output_dir = f"./outputs/pop_{utils.get_fid()}/"
utils.ensure_dir(output_dir)

log_dir = "../outputs"
fid = utils.get_fid()
log_file = f"../outputs/danna_practical_{fid}.txt"
log_folder_flows = f"../outputs/danna_practical_{fid}/"

num_paths_per_flow = 16

danna_files = [
    ("../outputs/danna_practical_2022_08_16_17_13_19_dc44cf7e", 0),
    ("../outputs/danna_practical_2022_08_16_22_22_21_6afe1d48", 0),
    ("../outputs/danna_practical_2023_04_03_03_53_50_cf6c522a", 0),
    ("../outputs/danna_practical_2023_04_04_07_24_57_37a821ce", 0)
]

output_fairness_baseline = benchmark_plot_utils.read_rate_log_file(constants.DANNA, danna_files,
                                                                   "*", {constants.DANNA: "*"},
                                                                   {})

danna_fid_to_rate_mapping = output_fairness_baseline[2]

for TOPO_NAME in TOPO_NAME_LIST:
    for TM_MODEL, DECIMAL_POINT in TM_NUMERIC_MODEL_LIST:
        configuration_params = constants.TOPOLOGY_TO_APPROX_BET_MCF_PARAMS[TOPO_NAME]
        num_bins, min_epsilon, min_beta, k, link_cap_scale_factor, num_iter_approx, num_iter_bet, base_split = configuration_params
        pop_client_split_fraction = constants.POP_TM_TYPE_TO_CLIENT_SPLIT_FRACTION[TM_MODEL]

        for SCALE_FACTOR in SCALE_FACTOR_LIST:
            per_topo_tm_output_dir = output_dir + f"topo_{TOPO_NAME}/tm_{TM_MODEL}/sf_{SCALE_FACTOR}/"
            utils.ensure_dir(per_topo_tm_output_dir)
            per_topo_tm_output_file = per_topo_tm_output_dir + "log.txt"

            fnames = utils.find_topo_tm_scale_factor_fname(TOPO_NAME, TM_MODEL, SCALE_FACTOR)

            rng_problem_chooser = np.random.RandomState(seed=0)
            fnames_idx = rng_problem_chooser.choice(len(fnames), num_scenario_per_topo_traffic)

            for selected_fidx in fnames_idx:
                file_name = fnames[selected_fidx]
                traffic_name_init = file_name[4].split("/")[-1]
            # for traffic_name_init in traffic_name_init_list:
                log = f"************** {traffic_name_init} ******************\n"
                print(log)
                utils.write_to_file(log, per_topo_tm_output_dir, per_topo_tm_output_file)

                # found = False
                # for file_name in fnames:
                #     if traffic_name_init in file_name[4]:
                #         found = True
                #         break
                # assert found
                problem = Problem.from_file(file_name[3], file_name[4])
                link_cap = 1000.0

                path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                                    k=num_paths_per_flow, number_processors=32,
                                                                                    base_split=base_split,
                                                                                    split_type=split_type)
                path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output

                if DECIMAL_POINT > 0:
                    print(f"======== rounding demands to {DECIMAL_POINT} decimal points")
                    problem = utils.round_demands(problem, DECIMAL_POINT)
                utils.revise_list_commodities(problem)

                try:
                    danna_fid_to_rate = utils.read_pickle_file(danna_fid_to_rate_mapping[num_paths_per_flow][f"'{file_name[4]}'"])
                except KeyError:
                    danna_output = danna_practical_max_min_fair.get_rates(0.1,
                                                                          problem,
                                                                          path_dict,
                                                                          link_cap=link_cap,
                                                                          feasibility_grb_method=2,
                                                                          mcf_grb_method=2,
                                                                          numeric_num_decimals=DECIMAL_POINT)
                    per_flow_log_file_name = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths_per_flow}"
                    danna_practical.store_danna_logs(log_dir, log_file, log_folder_flows, num_paths_per_flow, file_name,
                                                     per_flow_log_file_name, danna_output)
                    danna_fid_to_rate = danna_output[1]

                danna_to_fid_rate_vector = np.zeros(shape=len(problem.sparse_commodity_list))
                for fid, (src, dst, _) in problem.sparse_commodity_list:
                    danna_to_fid_rate_vector[fid] = np.sum(danna_fid_to_rate[src, dst])

                for num_sub_problems in num_sub_problems_list:
                    # popv2 + eb
                    popv2_eb_output = popv2_equi_binner.get_rates(num_sub_problems, pop_client_split_fraction,
                                                                  problem, path_dict, path_edge_idx_mapping,
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
                                                                  link_cap_scale_factor=link_cap_scale_factor,
                                                                  num_grb_threads=num_threads)
                    popv2_eb_fid_to_flow_rate_mapping, popv2_eb_duration, popv2_eb_run_time_dict = popv2_eb_output

                    popv2_eb_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                                        popv2_eb_fid_to_flow_rate_mapping,
                                                                                                        problem.sparse_commodity_list,
                                                                                                        theta_fairness=0.1)

                    popv2_eb_total_flow = 0
                    for (src, dst) in popv2_eb_fid_to_flow_rate_mapping:
                        popv2_eb_total_flow += sum(popv2_eb_fid_to_flow_rate_mapping[src, dst])

                    popv2_effective_run_time = benchmark_plot_utils.get_output_pop_run_time(constants.APPROX_BET_MCF,
                                                                                            popv2_eb_run_time_dict,
                                                                                            [popv2_equi_binner.POP_SPLIT],
                                                                                            [popv2_equi_binner.POP_SUB_PROBLEM],
                                                                                            num_sub_problems,
                                                                                            constants.approach_to_valid_for_run_time)

                    log = f"======== aproach {num_sub_problems}x pop v2 eb \n" + \
                        f"run time dict: {popv2_eb_run_time_dict} \n" + \
                        f"effective run time: {popv2_effective_run_time} \n" + \
                        f"total flow: {popv2_eb_total_flow} \n" + \
                        f"fairness: {popv2_eb_fairness_no} \n"
                    print(log)
                    utils.write_to_file(log, per_topo_tm_output_dir, per_topo_tm_output_file)

                # for num_sub_problems in num_sub_problems_list:
                #     # popv1 + eb
                #     popv1_eb_output = popv1_equi_binner.get_rates(num_sub_problems, problem, path_dict, path_edge_idx_mapping,
                #                                                   path_path_len_mapping, path_total_path_len_mapping,
                #                                                   path_split_ratio_mapping, num_paths_per_flow,
                #                                                   num_iter_approx_water=num_iter_approx,
                #                                                   num_iter_bet=num_iter_bet,
                #                                                   link_cap=link_cap,
                #                                                   mcf_grb_method=2,
                #                                                   num_bins=num_bins,
                #                                                   min_beta=min_beta,
                #                                                   min_epsilon=min_epsilon,
                #                                                   k=k,
                #                                                   alpha=0,
                #                                                   link_cap_scale_factor=link_cap_scale_factor,
                #                                                   num_grb_threads=num_threads)
                #     popv1_eb_fid_to_flow_rate_mapping, popv1_eb_duration, popv1_eb_run_time_dict = popv1_eb_output

                #     popv1_eb_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                #                                                                                         popv1_eb_fid_to_flow_rate_mapping,
                #                                                                                         problem.sparse_commodity_list,
                #                                                                                         theta_fairness=0.1)

                #     popv1_eb_total_flow = 0
                #     for (src, dst) in popv1_eb_fid_to_flow_rate_mapping:
                #         popv1_eb_total_flow += sum(popv1_eb_fid_to_flow_rate_mapping[src, dst])

                #     log = f"======== aproach {num_sub_problems}x pop v1 eb \n" + \
                #         f"run time dict: {popv1_eb_run_time_dict} \n" + \
                #         f"total flow: {popv1_eb_total_flow} \n" + \
                #         f"fairness: {popv1_eb_fairness_no} \n"
                #     print(log)
                #     utils.write_to_file(log, per_topo_tm_output_dir, per_topo_tm_output_file)

                # eb
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
                                                  link_cap_scale_factor=link_cap_scale_factor,
                                                  num_grb_threads=num_threads)
                eb_fid_to_flow_rate_mapping, eb_duration, eb_run_time_dict = eb_output

                eb_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                              eb_fid_to_flow_rate_mapping,
                                                                                              problem.sparse_commodity_list,
                                                                                              theta_fairness=0.1)

                eb_total_flow = 0
                for (src, dst) in eb_fid_to_flow_rate_mapping:
                    eb_total_flow += sum(eb_fid_to_flow_rate_mapping[src, dst])

                eb_effective_run_time = benchmark_plot_utils.get_output_run_time(constants.APPROX_BET_MCF,
                                                                                 eb_run_time_dict,
                                                                                 constants.approach_to_valid_for_run_time)

                log = "======== aproach eb \n" + \
                    f"run time dict: {eb_run_time_dict} \n" + \
                    f"effective run time: {eb_effective_run_time} \n" + \
                    f"total flow: {eb_total_flow} \n" + \
                    f"fairness: {eb_fairness_no} \n"
                print(log)
                utils.write_to_file(log, per_topo_tm_output_dir, per_topo_tm_output_file)

                # pop + gb
                for num_sub_problems in num_sub_problems_list:
                    alpha = 2
                    U = 0.1
                    max_epsilon = 1e-6
                    link_cap_multiplier = 1
                    pop_gb_output = pop_geometric_approx_binning.max_min_approx(num_sub_problems, pop_client_split_fraction,
                                                                                alpha, U, problem, path_dict, link_cap,
                                                                                max_epsilon, break_down=False, link_cap_scale_multiplier=link_cap_multiplier,
                                                                                mcf_grb_method=mcf_grb_method,
                                                                                num_grb_threads=num_threads)
                    pop_gb_fid_to_flow_rate_mapping, pop_db_dur, pop_gb_run_time_dict = pop_gb_output

                    pop_gb_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                                      pop_gb_fid_to_flow_rate_mapping,
                                                                                                      problem.sparse_commodity_list,
                                                                                                      theta_fairness=0.1)

                    pop_gb_total_flow = 0
                    for (src, dst) in pop_gb_fid_to_flow_rate_mapping:
                        pop_gb_total_flow += sum(pop_gb_fid_to_flow_rate_mapping[src, dst])

                    pop_gb_effective_run_time = benchmark_plot_utils.get_output_pop_run_time(constants.NEW_APPROX,
                                                                                             pop_gb_run_time_dict,
                                                                                             [pop_geometric_approx_binning.POP_SPLIT],
                                                                                             [pop_geometric_approx_binning.POP_SUB_PROBLEM],
                                                                                             num_sub_problems,
                                                                                             constants.approach_to_valid_for_run_time)
                    log = f"======== aproach {num_sub_problems}x pop gb \n" + \
                        f"run time dict: {pop_gb_run_time_dict} \n" + \
                        f"effective run time: {pop_gb_effective_run_time} \n" + \
                        f"total flow: {pop_gb_total_flow} \n" + \
                        f"fairness: {pop_gb_fairness_no} \n"
                    print(log)
                    utils.write_to_file(log, per_topo_tm_output_dir, per_topo_tm_output_file)

                # gb
                alpha = 2
                U = 0.1
                max_epsilon = 1e-6
                link_cap_multiplier = 1
                gb_output = geometric_approx_binning.max_min_approx(alpha, U, problem, path_dict, link_cap,
                                                                    max_epsilon, break_down=False, link_cap_scale_multiplier=link_cap_multiplier,
                                                                    mcf_grb_method=mcf_grb_method, num_grb_threads=num_threads)
                gb_fid_to_flow_rate_mapping, gb_duration, gb_run_time_dict = gb_output

                gb_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                              gb_fid_to_flow_rate_mapping,
                                                                                              problem.sparse_commodity_list,
                                                                                              theta_fairness=0.1)
                gb_total_flow = 0
                for (src, dst) in gb_fid_to_flow_rate_mapping:
                    gb_total_flow += sum(gb_fid_to_flow_rate_mapping[src, dst])

                gb_effective_run_time = benchmark_plot_utils.get_output_run_time(constants.NEW_APPROX,
                                                                                 gb_run_time_dict,
                                                                                 constants.approach_to_valid_for_run_time)

                log = "======== aproach gb \n" + \
                    f"run time dict: {gb_run_time_dict} \n" + \
                    f"effective run time: {gb_effective_run_time} \n" + \
                    f"total flow: {gb_total_flow} \n" + \
                    f"fairness: {gb_fairness_no} \n"
                print(log)
                utils.write_to_file(log, per_topo_tm_output_dir, per_topo_tm_output_file)

                # pop + swan
                for num_sub_problems in num_sub_problems_list:
                    alpha = 2
                    U = 0.1
                    pop_swan_output = pop_swan.max_min_approx(num_sub_problems, pop_client_split_fraction,
                                                              alpha, U, problem, path_dict, link_cap,
                                                              mcf_grb_method=mcf_grb_method,
                                                              break_down=False, num_grb_threads=num_threads)
                    pop_swan_fid_to_flow_rate_mapping, pop_swan_dur, pop_swan_run_time_dict = pop_swan_output
                    pop_swan_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                                        pop_swan_fid_to_flow_rate_mapping,
                                                                                                        problem.sparse_commodity_list,
                                                                                                        theta_fairness=0.1)

                    pop_swan_total_flow = 0
                    for (src, dst) in pop_swan_fid_to_flow_rate_mapping:
                        pop_swan_total_flow += sum(pop_swan_fid_to_flow_rate_mapping[src, dst])

                    pop_swan_effective_run_time = benchmark_plot_utils.get_output_pop_run_time(constants.SWAN,
                                                                                               pop_swan_run_time_dict,
                                                                                               [pop_swan.POP_SPLIT],
                                                                                               [pop_swan.POP_SUB_PROBLEM],
                                                                                               num_sub_problems,
                                                                                               constants.approach_to_valid_for_run_time)

                    log = f"======== aproach {num_sub_problems}x pop swan \n" + \
                        f"run time dict: {pop_swan_run_time_dict} \n" + \
                        f"effective run time: {pop_swan_effective_run_time} \n" + \
                        f"total flow: {pop_swan_total_flow} \n" + \
                        f"fairness: {pop_swan_fairness_no} \n"
                    print(log)
                    utils.write_to_file(log, per_topo_tm_output_dir, per_topo_tm_output_file)

                # swan
                mcf_grb_method = 2
                alpha = 2
                U = 0.1
                swan_output = swan_max_min_approx.max_min_approx(alpha, U, problem, path_dict, link_cap,
                                                                 mcf_grb_method=mcf_grb_method,
                                                                 break_down=False, num_grb_threads=num_threads)
                swan_fid_to_flow_rate_mapping, swan_dur, swan_run_time_dict = swan_output
                swan_fairness_no = benchmark_plot_utils.compute_fairness_no_vectorized_baseline(danna_to_fid_rate_vector,
                                                                                                swan_fid_to_flow_rate_mapping,
                                                                                                problem.sparse_commodity_list,
                                                                                                theta_fairness=0.1)

                swan_total_flow = 0
                for (src, dst) in swan_fid_to_flow_rate_mapping:
                    swan_total_flow += sum(swan_fid_to_flow_rate_mapping[src, dst])

                swan_effective_run_time = benchmark_plot_utils.get_output_run_time(constants.SWAN,
                                                                                   swan_run_time_dict,
                                                                                   constants.approach_to_valid_for_run_time)
                log = "======== aproach swan \n" + \
                    f"run time dict: {swan_run_time_dict} \n" + \
                    f"effective run time: {swan_effective_run_time} \n" + \
                    f"total flow: {swan_total_flow} \n" + \
                    f"fairness: {swan_fairness_no} \n"
                print(log)
                utils.write_to_file(log, per_topo_tm_output_dir, per_topo_tm_output_file)
