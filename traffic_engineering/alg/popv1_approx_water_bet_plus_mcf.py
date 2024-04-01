from datetime import datetime

import numpy as np
from gurobi import *

from alg import approx_water_bet, approx_water_plus_mcf, waterfilling_utils
from ncflow.lib import problem
from POP.traffic_engineering.lib.partitioning.pop import entity_splitting

MODEL_TIME = "model_approx_bet_p_mcf"
EXTRACT_RATE = "extract_rate_approx_bet_p_mcf"
POP_SUB_PROBLEM = 'pop_subproblem'


def get_rates(num_subproblems, problem: problem.Problem, paths, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
              path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet, link_cap,
              mcf_grb_method=2, num_bins=5, min_beta=1e-4, min_epsilon=1e-2, k=1, alpha=0, link_cap_scale_factor=1,
              break_down=False, biased_toward_low_flow_rate=False, biased_alpha=None, return_heuristic_output=None, num_grb_threads=0):

    run_time_dict = dict()
    run_time_dict[MODEL_TIME] = 0
    run_time_dict[EXTRACT_RATE] = 0
    run_time_dict[POP_SUB_PROBLEM] = dict()

    st_time = datetime.now()
    flow_details = problem.sparse_commodity_list
    path_characteristics = waterfilling_utils.get_vectorized_path_characteristics(problem,
                                                                                  path_edge_idx_mapping,
                                                                                  path_path_len_mapping,
                                                                                  num_paths_per_flow)

    _, _, demand_vector, _, _ = path_characteristics

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
    if return_heuristic_output:
        heuristic_output = flow_rate_matrix.copy()
    for key, val in run_time_dict_1.items():
        assert key not in run_time_dict
        run_time_dict[key] = val

    if break_down:
        checkpoint1 = datetime.now()

    if all_satisfied:
        output = flow_rate_matrix
    else:
        model_st_time = datetime.now()
        num_flows = len(flow_details)
        output_split = _split_flows(num_subproblems, paths, flow_details, flow_rate_matrix, num_flows)
        part_list_possible_paths, part_link_src_dst_path_dict, part_flow_details, part_to_sorted_fids = output_split

        if break_down:
            checkpoint2 = datetime.now()

        # Setting up the parameters
        num_flows_per_barrier = np.int(np.ceil(num_flows / (num_subproblems * num_bins)))
        if num_bins > 1:
            beta = np.power(min_beta, 1 / (num_bins - 1))
            epsilon = np.power(min_epsilon, 1 / (num_bins - 1))
        else:
            beta = 0
            epsilon = 1
        run_time_dict[MODEL_TIME] += (datetime.now() - model_st_time).total_seconds()

        output = [dict()]
        for part_id in range(num_subproblems):
            link_src_dst_path_dict = part_link_src_dst_path_dict[part_id]
            list_possible_paths = part_list_possible_paths[part_id]
            part_to_sorted_fids_wo_padding = part_to_sorted_fids[part_id][part_to_sorted_fids[part_id] >= 0]
            part_flow_rate_matrix = flow_rate_matrix[part_to_sorted_fids_wo_padding, :]
            part_demand_vector = demand_vector[part_to_sorted_fids_wo_padding]
            part_output = approx_water_plus_mcf.compute_throughput_path_based_given_tm(part_flow_details[part_id], part_flow_rate_matrix,
                                                                                       link_cap / num_subproblems, part_demand_vector,
                                                                                       link_src_dst_path_dict, list_possible_paths,
                                                                                       epsilon=epsilon, k=k, alpha=alpha, beta=beta,
                                                                                       num_flows_per_barrier=num_flows_per_barrier,
                                                                                       mcf_grb_method=mcf_grb_method,
                                                                                       break_down=break_down,
                                                                                       link_cap_scale_multiplier=link_cap_scale_factor,
                                                                                       num_paths_per_flow=num_paths_per_flow,
                                                                                       num_grb_threads=num_grb_threads // num_subproblems)
            run_time_dict_2 = part_output[-1]
            part_output = part_output[:-1]
            run_time_dict[POP_SUB_PROBLEM][part_id] = dict()
            for key, val in run_time_dict_2.items():
                assert key not in run_time_dict
                run_time_dict[POP_SUB_PROBLEM][part_id][key] = val

            for part_fid, fid in enumerate(part_to_sorted_fids_wo_padding):
                output[0][fid] = part_output[0][part_fid]

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

    if return_heuristic_output:
        heuristic_fid_rate_mapping = dict()
        for fid, (src, dst, _) in flow_details:
            heuristic_fid_rate_mapping[src, dst] = heuristic_output[fid]
        return fid_rate_mapping, heuristic_output, dur, run_time_dict
    print(f"approx bet mcf {dur}")
    print(f"approx bet mcf detail {run_time_dict}")
    return fid_rate_mapping, dur, run_time_dict


def _split_flows(num_subproblems, paths, flow_details, flow_rate_matrix, num_flows):
    throughput_per_flow = np.add.reduce(flow_rate_matrix, axis=1)
    sorted_fids = np.argsort(throughput_per_flow)
    if num_flows % num_subproblems > 0:
        sorted_fids = np.pad(sorted_fids, (0, num_subproblems - num_flows % num_subproblems), constant_values=-1)
    part_list_possible_paths = [tuplelist() for _ in range(num_subproblems)]
    part_link_src_dst_path_dict = [tupledict() for _ in range(num_subproblems)]
    part_flow_details = [[] for _ in range(num_subproblems)]
    part_to_sorted_fids = np.transpose(np.split(sorted_fids, int(np.ceil(num_flows / num_subproblems))))
    for fid, (src, dst, demand) in flow_details:
        if src == dst:
            continue
        part_id = np.where(sorted_fids == fid)[0][0] % num_subproblems
        part_fid = np.where(part_to_sorted_fids[part_id] == fid)[0][0]
        part_flow_details[part_id].append((part_fid, (src, dst, demand)))
        index = 0
        for path in paths[(src, dst)]:
            part_list_possible_paths[part_id].append((part_fid, index))
            for i in range(1, len(path)):
                if (path[i - 1], path[i]) not in part_link_src_dst_path_dict[part_id]:
                    part_link_src_dst_path_dict[part_id][path[i - 1], path[i]] = list()
                part_link_src_dst_path_dict[part_id][path[i - 1], path[i]].append((part_fid, index))
            index += 1
    return part_list_possible_paths, part_link_src_dst_path_dict, part_flow_details, part_to_sorted_fids


def _split(num_subproblems, split_fraction, paths, flow_details, flow_rate_matrix, num_flows):
    throughput_per_flow = np.add.reduce(flow_rate_matrix, axis=1)

    sub_problems = [problem.copy() for _ in range(num_subproblems)]
    # zero-out the traffic matrices; they will be populated at random using commodity list
    for sp in sub_problems:
        for u in sp.G.nodes:
            for v in sp.G.nodes:
                sp.traffic_matrix.tm[u, v] = 0

    entity_list = [[k, u, v, throughput_per_flow[k]] for (k, (u, v, _)) in problem.commodity_list]

    split_entity_lists = entity_splitting.split_entities(entity_list, split_fraction)
    for split_list in split_entity_lists:

        num_subentities = len(split_list)
        assigned_sps_list = []
        # create list of assigned sps by randomly sampling sps (without replacement, if possible)
        # until all entities have been assigned
        while len(assigned_sps_list) < num_subentities:
            num_to_add = min(
                [num_subentities - len(assigned_sps_list), num_subproblems]
            )
            randperm = np.random.permutation(np.arange(num_subproblems))
            assigned_sps_list += list(randperm[:num_to_add])

        for ind, [_, source, target, demand] in enumerate(split_list):
            sub_problems[assigned_sps_list[ind]].traffic_matrix.tm[
                source, target
            ] += demand

    for sub_problem in sub_problems:
        for u, v in sub_problems[-1].G.edges:
            sub_problem.G[u][v]["capacity"] = (
                sub_problem.G[u][v]["capacity"] / self._num_subproblems
            )
    return sub_problems