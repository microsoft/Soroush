import time
from datetime import datetime

import numpy as np
from gurobi import *

from alg import approx_water_bet, approx_water_plus_mcf, waterfilling_utils
from ncflow.lib import problem

MODEL_TIME = "model_approx_bet_p_mcf"
EXTRACT_RATE = "extract_rate_approx_bet_p_mcf"


def get_rates(problem: problem.Problem, paths, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
              path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet, link_cap,
              mcf_grb_method=2, num_bins=5, min_beta=1e-4, min_epsilon=1e-2, k=1, alpha=0, link_cap_scale_factor=1,
              break_down=False, biased_toward_low_flow_rate=False, biased_alpha=None):

    run_time_dict = dict()
    run_time_dict[MODEL_TIME] = 0
    run_time_dict[EXTRACT_RATE] = 0

    st_time = datetime.now()
    flow_details = problem.sparse_commodity_list
    path_characteristics = waterfilling_utils.get_vectorized_path_characteristics(problem,
                                                                                  path_edge_idx_mapping,
                                                                                  path_path_len_mapping,
                                                                                  num_paths_per_flow)

    sub_fid_path_len_vector, edge_idx_vector, demand_vector, path_exist_vector, num_sub_flows = path_characteristics

    run_time_dict[MODEL_TIME] = (datetime.now() - st_time).total_seconds()

    flow_rate_matrix, approx_dur, all_satisfied, run_time_dict_1 = approx_water_bet.get_rates(problem, path_split_ratio_mapping,
                                                                                              num_paths_per_flow,
                                                                                              num_iter_approx_water=num_iter_approx_water,
                                                                                              num_iter_bet=num_iter_bet,
                                                                                              link_cap=link_cap,
                                                                                              path_characteristics=path_characteristics,
                                                                                              break_down=break_down,
                                                                                              bias_toward_low_flow_rate=biased_toward_low_flow_rate,
                                                                                              bias_alpha=biased_alpha,
                                                                                              return_matrix=True)
    for key, val in run_time_dict_1.items():
        assert key not in run_time_dict
        run_time_dict[key] = val

    if break_down:
        checkpoint1 = datetime.now()

    if all_satisfied:
        output = flow_rate_matrix
    else:
        model_st_time = datetime.now()
        list_possible_paths = tuplelist()
        link_src_dst_path_dict = tupledict()
        for fid, (src, dst, demand) in flow_details:
            if src == dst:
                continue
            index = 0
            for path in paths[(src, dst)]:
                list_possible_paths.append((fid, index))
                for i in range(1, len(path)):
                    if (path[i - 1], path[i]) not in link_src_dst_path_dict:
                        link_src_dst_path_dict[path[i - 1], path[i]] = list()
                    link_src_dst_path_dict[path[i - 1], path[i]].append((fid, index))
                index += 1

        if break_down:
            checkpoint2 = datetime.now()

        # Setting up the parameters
        num_flows = len(flow_details)
        num_flows_per_barrier = np.int(np.ceil(num_flows / num_bins))
        beta = np.power(min_beta, 1/(num_bins - 1))
        epsilon = np.power(min_epsilon, 1/(num_bins - 1))
        run_time_dict[MODEL_TIME] += (datetime.now() - model_st_time).total_seconds()

        output = approx_water_plus_mcf.compute_throughput_path_based_given_tm(flow_details, flow_rate_matrix, link_cap, demand_vector,
                                                                              link_src_dst_path_dict, list_possible_paths,
                                                                              epsilon=epsilon, k=k, alpha=alpha, beta=beta,
                                                                              num_flows_per_barrier=num_flows_per_barrier,
                                                                              mcf_grb_method=mcf_grb_method,
                                                                              break_down=break_down,
                                                                              link_cap_scale_multiplier=link_cap_scale_factor,
                                                                              num_paths_per_flow=num_paths_per_flow)
        run_time_dict_2 = output[-1]
        output = output[:-1]
        for key, val in run_time_dict_2.items():
            assert key not in run_time_dict
            run_time_dict[key] = val

    if break_down:
        checkpoint3 = datetime.now()
        opt_detailed_dur = output[1]
        output = output[0]
        if all_satisfied:
            dur = ((checkpoint1 - st_time).total_seconds(), approx_dur)
        else:
            dur = (((checkpoint2 - checkpoint1).total_seconds(), (checkpoint3 - checkpoint2).total_seconds()), approx_dur, opt_detailed_dur)
    else:
        if not all_satisfied:
            output = output[0]
        dur = (datetime.now() - st_time).total_seconds()

    extract_rate_st_time = datetime.now()
    fid_rate_mapping = dict()
    for fid, (src, dst, _) in flow_details:
        fid_rate_mapping[src, dst] = output[fid]
    run_time_dict[EXTRACT_RATE] = (datetime.now() - extract_rate_st_time).total_seconds()

    print(f"approx bet mcf {dur}")
    print(f"approx bet mcf detail {run_time_dict}")
    return fid_rate_mapping, dur, run_time_dict
