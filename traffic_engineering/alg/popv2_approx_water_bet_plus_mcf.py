from datetime import datetime

from ncflow.lib import problem
from POP.traffic_engineering.lib.partitioning.pop import RandomSplitter
from alg import approx_water_bet_plus_mcf as eb_solver
from utilities import utils

POP_SPLIT = 'pop_split'
POP_SUB_PROBLEM = 'pop_subproblem'


def get_rates(num_sub_problems, pop_split_fraction, problem: problem.Problem, paths, path_edge_idx_mapping,
              path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping, num_paths_per_flow,
              num_iter_approx_water, num_iter_bet, link_cap, mcf_grb_method=2, num_bins=5, min_beta=1e-4, min_epsilon=1e-2,
              k=1, alpha=0, link_cap_scale_factor=1, break_down=False, biased_toward_low_flow_rate=False,
              biased_alpha=None, return_heuristic_output=None, num_grb_threads=0):
    run_time_dict = dict()
    run_time_dict[POP_SPLIT] = 0
    run_time_dict[POP_SUB_PROBLEM] = dict()
    dur = []
    flow_id_to_rate_mapping = dict()

    st_time = datetime.now()
    sub_problems = _split_flows_to_partitions(num_sub_problems, pop_split_fraction, problem)
    run_time_dict[POP_SPLIT] = (datetime.now() - st_time).total_seconds()

    for part_id, sub_problem in enumerate(sub_problems):
        sub_problem._invalidate_commodity_lists()
        utils.revise_list_commodities(sub_problem)
        part_eb_output = eb_solver.get_rates(sub_problem, paths, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
                                             path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet,
                                             link_cap / num_sub_problems, mcf_grb_method, num_bins, min_beta, min_epsilon, k, alpha,
                                             link_cap_scale_factor, break_down, biased_toward_low_flow_rate, biased_alpha,
                                             return_heuristic_output, num_grb_threads=num_grb_threads // num_sub_problems)
        part_eb_id_to_rate, part_dur, part_run_time_dict = part_eb_output
        run_time_dict[POP_SUB_PROBLEM][part_id] = part_run_time_dict
        dur.append(part_dur)

        for (src, dst) in part_eb_id_to_rate:
            # assert (src, dst) not in flow_id_to_rate_mapping
            if (src, dst) not in flow_id_to_rate_mapping:
                flow_id_to_rate_mapping[src, dst] = part_eb_id_to_rate[src, dst]
            else:
                assert pop_split_fraction > 0
                flow_id_to_rate_mapping[src, dst] += part_eb_id_to_rate[src, dst]

    return flow_id_to_rate_mapping, dur, run_time_dict


def _split_flows_to_partitions(num_sub_problems, pop_split_fraction, problem: problem.Problem):
    pop_splitter = RandomSplitter(num_sub_problems, pop_split_fraction)
    return pop_splitter.split(problem)
