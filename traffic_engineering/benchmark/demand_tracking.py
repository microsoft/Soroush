#!/usr/bin/env python

from glob import glob
import numpy as np

import sys

sys.path.append('..')

from alg import danna_practical_max_min_fair as danna_solver
from alg import approx_water_bet_plus_mcf as eb_solver
from alg import approx_water_bet as heuristic_solver
from alg import swan_max_min_approx as swan_solver
from scripts import benchmark_plot_utils
from utilities import constants, utils, shortest_paths
from ncflow.lib import Problem


TOPOLOGY = 'Cogentco.graphml'
TM_MODEL = 'uniform'
SCALE_FACTOR = 32.0


def generate_sequence_of_tms(seed_prob,
                             num_tms,
                             rel_delta_abs_mean,
                             rel_delta_std,
                             traffic_format,
                             seed=1):
    problems = [seed_prob]
    mean_load = np.mean(seed_prob.traffic_matrix.tm)
    # The mean and std arguments are relative to this traffic matrix;
    # we scale them by the actual mean of the seed TM
    delta_mean = rel_delta_abs_mean * mean_load
    delta_std = rel_delta_std * mean_load

    np.random.seed(seed)
    curr_idx = 1
    while len(problems) < num_tms:
        traffic_dir = traffic_format
        traffic_file = traffic_dir + f"/traffic_{curr_idx}.pickle"
        if utils.file_exists(traffic_file):
            # load existing traffic
            new_prob = utils.read_pickle_file(traffic_file)
        else:
            # create new traffic
            new_prob = problems[-1].copy()
            # Change each demand by delta mean on average, with a spread of delta_std
            perturb_mean = np.random.normal(delta_mean, delta_std)
            perturb_mean *= np.random.choice([-1, 1])
            new_prob.traffic_matrix.perturb_matrix(perturb_mean, delta_std)
            utils.ensure_dir(traffic_dir)
            utils.write_to_file_as_pickle(new_prob, traffic_dir, traffic_file)
        problems.append(new_prob)
        curr_idx += 1
    return problems


class Danna(object):
    def __init__(self):
        self.solver = danna_solver
        self.feasibility_grb_method = 1
        self.mcf_grb_method = 2
        self.approach_name = constants.DANNA
        self.run_time = None
        self.fid_to_per_path_rate = None

    def solve(self, problem, path_output, num_paths_per_flow, link_cap):
        path_dict, _, _, _, _ = path_output
        output = self.solver.get_rates(0, problem, paths=path_dict, link_cap=link_cap,
                                       feasibility_grb_method=self.feasibility_grb_method,
                                       mcf_grb_method=self.mcf_grb_method)
        fid_to_total_rate, fid_to_per_path_rate, dur, run_time_dict = output
        effective_dur = benchmark_plot_utils.get_output_run_time(self.approach_name,
                                                                 run_time_dict,
                                                                 constants.approach_to_valid_for_run_time)
        self.run_time = effective_dur
        self.fid_to_per_path_rate = fid_to_per_path_rate

    @property
    def sol_dict(self):
        return self.fid_to_per_path_rate

    @property
    def name(self):
        return self.approach_name

    @property
    def runtime(self):
        return self.run_time


class INSTANTDANNA(Danna):
    @property
    def runtime(self):
        return 0


class EB(object):
    def __init__(self, topo_name):
        self.approach_name = constants.APPROX_BET_MCF
        self.run_time = None
        self.fid_to_per_path_rate = None
        self.num_bins, self.min_epsilon, self.min_beta, self.k, self.link_cap_scale_factor, self.num_iter_approx, self.num_iter_bet, self.base_split = \
            constants.TOPOLOGY_TO_APPROX_BET_MCF_PARAMS[topo_name]
        self.mcf_grb_method = 2

    def solve(self, problem, path_output, num_paths_per_flow, link_cap):
        path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output
        output = eb_solver.get_rates(problem, path_dict,
                                     path_edge_idx_mapping,
                                     path_path_len_mapping,
                                     path_total_path_len_mapping,
                                     path_split_ratio_mapping,
                                     num_paths_per_flow,
                                     num_iter_approx_water=self.num_iter_approx,
                                     num_iter_bet=self.num_iter_bet,
                                     link_cap=link_cap,
                                     mcf_grb_method=self.mcf_grb_method,
                                     num_bins=self.num_bins,
                                     min_beta=self.min_beta,
                                     min_epsilon=self.min_epsilon,
                                     k=self.k,
                                     alpha=0,
                                     link_cap_scale_factor=self.link_cap_scale_factor)
        fid_to_flow_rate_mapping, dur, run_time_dict = output
        self.run_time = benchmark_plot_utils.get_output_run_time(self.approach_name, run_time_dict,
                                                                 constants.approach_to_valid_for_run_time)
        self.fid_to_flow_rate = fid_to_flow_rate_mapping

    @property
    def sol_dict(self):
        return self.fid_to_flow_rate

    @property
    def name(self):
        return self.approach_name

    @property
    def runtime(self):
        return self.run_time


class Heuristic(object):
    def __init__(self, topo_name):
        self.approach_name = constants.APPROX_BET
        self.run_time = None
        self.fid_to_per_path_rate = None
        _, _, _, _, _, self.num_iter_approx, self.num_iter_bet, _ = constants.TOPOLOGY_TO_APPROX_BET_MCF_PARAMS[topo_name]

    def solve(self, problem, path_output, num_paths_per_flow, link_cap):
        path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output
        output = heuristic_solver.get_rates(problem,
                                            path_split_ratio_mapping,
                                            num_paths_per_flow,
                                            num_iter_approx_water=self.num_iter_approx,
                                            num_iter_bet=self.num_iter_bet,
                                            link_cap=link_cap,
                                            path_edge_idx_mapping=path_edge_idx_mapping,
                                            path_path_len_mapping=path_path_len_mapping,
                                            bias_toward_low_flow_rate=False)
        fid_to_flow_rate_mapping, dur, _, run_time_dict = output
        self.run_time = benchmark_plot_utils.get_output_run_time(self.approach_name, run_time_dict,
                                                                 constants.approach_to_valid_for_run_time)
        self.fid_to_flow_rate = fid_to_flow_rate_mapping

    @property
    def sol_dict(self):
        return self.fid_to_flow_rate

    @property
    def name(self):
        return self.approach_name

    @property
    def runtime(self):
        return self.run_time


class SWAN(object):
    def __init__(self):
        self.approach_name = constants.SWAN
        self.alpha = 2
        self.U = 0.1
        self.mcf_grb_method = 2

    def solve(self, problem, path_output, num_paths_per_flow, link_cap):
        path_dict, _, _, _, _ = path_output
        swan_output = swan_solver.max_min_approx(self.alpha, self.U, problem, path_dict, link_cap,
                                                 mcf_grb_method=self.mcf_grb_method,
                                                 break_down=False)
        fid_to_flow_rate_mapping, dur, _, run_time_dict = swan_output
        self.run_time = benchmark_plot_utils.get_output_run_time(self.approach_name, run_time_dict,
                                                                 constants.approach_to_valid_for_run_time)
        self.fid_to_flow_rate = fid_to_flow_rate_mapping

    @property
    def sol_dict(self):
        return self.fid_to_flow_rate

    @property
    def name(self):
        return self.approach_name

    @property
    def runtime(self):
        return self.run_time


def compute_satisfied_demand(problem, sol_dict, residual_factor):
    if sol_dict is None:
        return 0.0, problem.traffic_matrix.tm.copy() * residual_factor

    curr_demand_dict = {(s_k, t_k): d_k
                        for _, (s_k, t_k, d_k) in problem.commodity_list}
    assigned_rates = dict()
    residual_dict = dict()
    residual_tm = np.zeros_like(problem.traffic_matrix.tm)

    for (src, dst), flow_list in sol_dict.items():
        if (src, dst) not in curr_demand_dict:
            continue
        curr_demand = curr_demand_dict[src, dst]
        total_flow = np.sum(flow_list)
        # If the current problem requests less flow than what we solved for,
        # we only provide the flow we solved for
        real_flow = min(curr_demand, total_flow)
        assigned_rates[src, dst] = real_flow
        residual_demand = curr_demand - real_flow
        residual_tm[src, dst] = residual_demand
        residual_dict[src, dst] = residual_demand * residual_factor

    return assigned_rates, residual_dict, residual_tm * residual_factor


def demand_tracking(algo, problems, time_per_prob, residual_factor, path_output, num_paths_per_flow, link_cap):
    results_dirname = f"../demand_tracking_{algo.name}_{utils.get_fid()}/"
    utils.ensure_dir(results_dirname)

    i = 0
    curr_sol_dict = None
    while i < len(problems):
        problem = problems[i]
        print('=== Problem {}'.format(i))
        # self, problem, path_output, num_paths_per_flow, link_cap
        algo.solve(problem, path_output, num_paths_per_flow, link_cap)
        runtime = algo.runtime
        print(f'************ runtime {runtime}')
        while runtime > time_per_prob and i < len(problems):
            satisfied_demand, residual_dict, residual_tm = compute_satisfied_demand(
                problems[i], curr_sol_dict, residual_factor)
            utils.write_to_file_as_pickle(satisfied_demand, results_dirname, results_dirname + f"satisfied_{i}.pickle")
            utils.write_to_file_as_pickle(residual_dict, results_dirname, results_dirname + f"residual_{i}.pickle")
            runtime -= time_per_prob
            i += 1
            if i < len(problems):
                problems[i].traffic_matrix.tm += residual_tm

        if i >= len(problems):
            break
        curr_sol_dict = algo.sol_dict
        satisfied_demand, residual_dict, residual_tm = compute_satisfied_demand(
            problems[i], curr_sol_dict, residual_factor)
        utils.write_to_file_as_pickle(satisfied_demand, results_dirname, results_dirname + f"satisfied_{i}.pickle")
        utils.write_to_file_as_pickle(residual_dict, results_dirname, results_dirname + f"residual_{i}.pickle")
        i += 1
        if i >= len(problems):
            break
        problems[i].traffic_matrix.tm += residual_tm


def print_delta_norms(problems):
    first_prob = problems[0]
    prev_tm = first_prob.traffic_matrix.tm
    prev_norm = np.linalg.norm(prev_tm)

    delta_norm_norms = []
    for problem in problems[1:]:
        tm = problem.traffic_matrix.tm
        delta_tm = tm - prev_tm
        delta_norm = np.linalg.norm(delta_tm)
        delta_norm_norm = delta_norm / prev_norm
        print(delta_norm_norm)
        delta_norm_norms.append(delta_norm_norm)
        prev_tm = tm
        prev_norm = np.linalg.norm(tm)

    print('Mean:', np.mean(delta_norm_norms))


def print_total_demand(problems):
    total_demands = []
    for problem in problems:
        print(problem.total_demand)
        total_demands.append(problem.total_demand)

    print('Mean:', np.mean(total_demands))


def print_entry_demands(problems, nodelist):
    assert len(nodelist) == 2
    i = nodelist[0]
    j = nodelist[1]
    print("Demands for {}->{}".format(i, j))
    for problem in problems:
        print(problem.traffic_matrix.tm[i, j])


# self, problem, path_output, num_paths_per_flow, link_cap
if __name__ == '__main__':
    idx = 0
    seed_tm_fname = glob('../ncflow/traffic-matrices/{}/{}_{}_*_{}_*.pkl'.format(
        TM_MODEL, TOPOLOGY, TM_MODEL, SCALE_FACTOR))[idx]

    print(f"=========== {seed_tm_fname}")
    seed_prob = Problem.from_file(
        '../ncflow/topologies/topology-zoo/{}'.format(TOPOLOGY), seed_tm_fname)
    num_tms = 25
    time_per_prob = 5 * 60

    ALG_LIST = [
        INSTANTDANNA(),
        # Danna(),
        EB(TOPOLOGY),
        SWAN(),
    ]

    rel_delta_abs_mean = 0.35
    rel_delta_std = 0.12
    num_paths_per_flow = 16
    link_cap = 1000.0
    traffic_format = '../outputs/demand_tracking_traffic/' + TOPOLOGY + '_' + TM_MODEL + '_' + \
                     str(SCALE_FACTOR) + '_' + str(idx) + '_' + str(rel_delta_abs_mean) + '_' + \
                     str(rel_delta_std)

    ### numbers borrowed from NCFlow
    problems = generate_sequence_of_tms(seed_prob,
                                        num_tms,
                                        rel_delta_abs_mean=rel_delta_abs_mean,
                                        rel_delta_std=rel_delta_std,
                                        traffic_format=traffic_format)
    print_total_demand(problems)
    
    path_output = shortest_paths.all_pair_shortest_path_multi_processor(seed_prob.G, seed_prob.edge_idx,
                                                                        k=num_paths_per_flow, number_processors=32)

    for algo in ALG_LIST:
        print('Algo: {}'.format(algo.name))
        demand_tracking(algo, problems, time_per_prob, 1, path_output, num_paths_per_flow, link_cap)