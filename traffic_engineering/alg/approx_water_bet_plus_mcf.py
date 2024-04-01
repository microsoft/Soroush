import time
from datetime import datetime

import numpy as np
from gurobi import *

from alg import approx_water_bet, approx_water_plus_mcf, waterfilling_utils
from ncflow.lib import problem


def get_rates(problem: problem.Problem, paths, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
              path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet, link_cap, break_down=False,
              biased_toward_low_flow_rate=False, biased_alpha=None):

    st_time = datetime.now()
    flow_details = problem.sparse_commodity_list
    path_characteristics = waterfilling_utils.get_vectorized_path_characteristics(problem,
                                                                                  path_edge_idx_mapping,
                                                                                  path_path_len_mapping,
                                                                                  num_paths_per_flow)

    sub_fid_path_len_vector, edge_idx_vector, demand_vector, path_exist_vector, num_sub_flows = path_characteristics
    flow_rate_matrix, approx_dur, all_satisfied = approx_water_bet.get_rates(problem, path_split_ratio_mapping,
                                                                             num_paths_per_flow,
                                                                             num_iter_approx_water=num_iter_approx_water,
                                                                             num_iter_bet=num_iter_bet,
                                                                             link_cap=link_cap,
                                                                             path_characteristics=path_characteristics,
                                                                             break_down=break_down,
                                                                             bias_toward_low_flow_rate=biased_toward_low_flow_rate,
                                                                             bias_alpha=biased_alpha,
                                                                             return_matrix=True)

    if break_down:
        checkpoint1 = datetime.now()

    if all_satisfied:
        output = flow_rate_matrix
    else:
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

        # Below setting works for fairness
        num_bins = 5
        num_flows = len(flow_details)
        num_flows_per_barrier = np.int(np.ceil(num_flows / num_bins))
        beta = np.power(1e-4, 1/num_bins)
        k = 1
        alpha = 0
        epsilon = np.power(1e-2, 1/num_bins)
        # beta = 0
        # epsilon = np.power(link_cap * 1e-7, 1/num_flows)

        # Below setting works for 50% efficiency 50% fairness
        # num_bins = 5
        # num_flows = len(flow_details)
        # num_flows_per_barrier = np.int(np.ceil(num_flows / num_bins))
        # beta = np.power(link_cap * 1e-10, 1/num_bins)
        # epsilon = np.power(1e-2, 1/num_bins)
        # beta = 0.001
        # k = link_cap/np.power(beta, 2)
        # alpha = 0

        output = approx_water_plus_mcf.compute_throughput_path_based_given_tm(flow_details, flow_rate_matrix, link_cap, demand_vector,
                                                                              link_src_dst_path_dict, list_possible_paths,
                                                                              epsilon=epsilon, k=k, alpha=alpha, beta=beta,
                                                                              num_flows_per_barrier=num_flows_per_barrier,
                                                                              break_down=break_down, link_cap_scale_multiplier=link_cap)

    if break_down:
        checkpoint3 = datetime.now()
        opt_detailed_dur = output[1]
        output = output[0]
        if all_satisfied:
            dur = ((checkpoint1 - st_time).total_seconds(), approx_dur)
        else:
            dur = (((checkpoint2 - checkpoint1).total_seconds(), (checkpoint3 - checkpoint2).total_seconds()), approx_dur, opt_detailed_dur)
    else:
        dur = (datetime.now() - st_time).total_seconds()

    fid_rate_mapping = dict()
    for fid, (src, dst, _) in flow_details:
        fid_rate_mapping[src, dst] = output[fid]
    return fid_rate_mapping, dur
