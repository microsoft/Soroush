from datetime import datetime

from ncflow.lib import problem
from POP.traffic_engineering.lib.partitioning.pop import RandomSplitter
from alg import geometric_approx_binning as gb
from utilities import utils

POP_SPLIT = 'pop_split'
POP_SUB_PROBLEM = 'pop_subproblem'


def max_min_approx(num_sub_problems, pop_split_fraction, alpha, U, problem: problem.Problem, paths, link_cap, max_epsilon,
                   break_down=False, link_cap_scale_multiplier=1, mcf_grb_method=2, num_grb_threads=0):
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
        part_gb_output = gb.max_min_approx(alpha, U, sub_problem, paths, link_cap / num_sub_problems, max_epsilon,
                                           break_down=break_down, link_cap_scale_multiplier=link_cap_scale_multiplier,
                                           mcf_grb_method=mcf_grb_method, num_grb_threads=num_grb_threads // num_sub_problems)
        part_gb_id_to_rate, part_dur, part_run_time_dict = part_gb_output
        run_time_dict[POP_SUB_PROBLEM][part_id] = part_run_time_dict
        dur.append(part_dur)

        for (src, dst) in part_gb_id_to_rate:
            # assert (src, dst) not in flow_id_to_rate_mapping
            if (src, dst) not in flow_id_to_rate_mapping:
                flow_id_to_rate_mapping[src, dst] = part_gb_id_to_rate[src, dst]
            else:
                assert pop_split_fraction > 0
                flow_id_to_rate_mapping[src, dst] += part_gb_id_to_rate[src, dst]

    return flow_id_to_rate_mapping, dur, run_time_dict


def _split_flows_to_partitions(num_sub_problems, pop_split_fraction, problem: problem.Problem):
    pop_splitter = RandomSplitter(num_sub_problems, pop_split_fraction)
    return pop_splitter.split(problem)
