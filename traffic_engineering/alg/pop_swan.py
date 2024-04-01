from datetime import datetime

from ncflow.lib import problem
from POP.traffic_engineering.lib.partitioning.pop import RandomSplitter
from alg import swan_max_min_approx as swan
from utilities import utils

POP_SPLIT = 'pop_split'
POP_SUB_PROBLEM = 'pop_subproblem'


def max_min_approx(num_sub_problems, pop_split_fraction, alpha, U, problem: problem.Problem, paths, link_cap,
                   mcf_grb_method=2, break_down=False, num_grb_threads=0):
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
        part_swan_output = swan.max_min_approx(alpha, U, sub_problem, paths, link_cap / num_sub_problems, mcf_grb_method,
                                               break_down, num_grb_threads // num_sub_problems)
        part_swan_id_to_rate, part_dur, part_run_time_dict = part_swan_output
        run_time_dict[POP_SUB_PROBLEM][part_id] = part_run_time_dict
        dur.append(part_dur)

        for (src, dst) in part_swan_id_to_rate:
            # assert (src, dst) not in flow_id_to_rate_mapping
            # flow_id_to_rate_mapping[src, dst] = part_swan_id_to_rate[src, dst]
            if (src, dst) not in flow_id_to_rate_mapping:
                flow_id_to_rate_mapping[src, dst] = part_swan_id_to_rate[src, dst]
            else:
                assert pop_split_fraction > 0
                flow_id_to_rate_mapping[src, dst] += part_swan_id_to_rate[src, dst]

    return flow_id_to_rate_mapping, dur, run_time_dict


def _split_flows_to_partitions(num_sub_problems, pop_split_fraction, problem: problem.Problem):
    pop_splitter = RandomSplitter(num_sub_problems, pop_split_fraction)
    return pop_splitter.split(problem)
