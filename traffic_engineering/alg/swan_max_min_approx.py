from collections import defaultdict
from datetime import datetime

import numpy as np
from gurobi import *

from utilities import constants
from ncflow.lib import problem

MODEL_TIME = 'model'
MCF_TOTAL = 'mcf_total'
MCF_SOLVER = 'mcf_solver'
FREEZE_TIME = 'freeze_time'


def max_min_approx(alpha, U, problem:problem.Problem, paths, link_cap, mcf_grb_method=2, break_down=False):
    run_time_dict = dict()
    run_time_dict[MODEL_TIME] = 0
    run_time_dict[MCF_TOTAL] = 0
    run_time_dict[MCF_SOLVER] = 0
    run_time_dict[FREEZE_TIME] = 0
    st_time = datetime.now()
    max_demand = 0
    min_demand = np.inf
    src_dst_demand_mapping = dict()
    flow_id_to_rate_mapping = dict()

    list_possible_paths = tuplelist()
    link_src_dst_path_dict = tupledict()
    index = 0
    frozen_flows = dict()

    for _, (src, dst, demand) in problem.sparse_commodity_list:
        max_demand = max(max_demand, demand)
        min_demand = min(min_demand, demand)
        src_dst_demand_mapping[src, dst] = demand
        if src == dst:
            continue
        for path in paths[(src, dst)]:
            list_possible_paths.append((src, dst, index))
            for i in range(1, len(path)):
                if (path[i - 1], path[i]) not in link_src_dst_path_dict:
                    link_src_dst_path_dict[path[i - 1], path[i]] = list()
                link_src_dst_path_dict[path[i - 1], path[i]].append((src, dst, index))
            index += 1
        # if demand < U:
        #     frozen_flows[src, dst] = demand

    # U = min(min_demand, U)
    U = max(U, link_cap / len(problem.sparse_commodity_list))
    T = max(1, int(np.ceil(np.log(max_demand / U) / np.log(alpha))))
    model, flow_vars, constraint_dict = create_initial_mcf_model(problem, link_cap, link_src_dst_path_dict,
                                                                 list_possible_paths, frozen_flows, mcf_grb_method)

    if break_down:
        checkpoint_1_time = datetime.now()
        time_freeze = 0
        time_create_model = 0
        time_solve_optimization = 0
        time_retrieve_values = 0
    run_time_dict[MODEL_TIME] = (datetime.now() - st_time).total_seconds()

    for k in range(T + 1):
        if k == 0:
            lb = 0
            ub = U
        else:
            lb = np.power(alpha, k - 1) * U
            ub = np.power(alpha, k) * U
        print(f"swan iteration {k}: time {(datetime.now() - st_time).total_seconds()} num frozen flows {len(frozen_flows)} "
              f"lb {lb} ub {ub}")

        output = compute_throughput_path_based_given_tm(problem, lb, ub, frozen_flows, model, constraint_dict, flow_vars,
                                                        run_time_dict, break_down=break_down)

        if break_down:
            flow_rate_mapping, t_m, t_o, t_r = output
            time_create_model += t_m
            time_solve_optimization += t_o
            time_retrieve_values += t_r
        else:
            flow_rate_mapping = output

        st_time_freeze = datetime.now()
        for (src, dst) in flow_rate_mapping:
            if (src, dst) not in frozen_flows:
                total_rate = sum(flow_rate_mapping[src, dst])
                if total_rate < min(src_dst_demand_mapping[src, dst], ub) - constants.O_epsilon or \
                        np.abs(total_rate - src_dst_demand_mapping[src, dst]) <= constants.O_epsilon:
                    frozen_flows[src, dst] = total_rate
                    modify_model(model, src, dst, total_rate, flow_vars, constraint_dict)
                    flow_id_to_rate_mapping[src, dst] = flow_rate_mapping[src, dst]
            else:
                flow_id_to_rate_mapping[src, dst] = flow_rate_mapping[src, dst]
        run_time_dict[FREEZE_TIME] += (datetime.now() - st_time_freeze).total_seconds()
        if break_down:
            time_freeze += (datetime.now() - st_time_freeze).total_seconds()

    if break_down:
        dur = ((checkpoint_1_time - st_time).total_seconds(), time_create_model, time_solve_optimization, time_retrieve_values, time_freeze)
    else:
        dur = (datetime.now() - st_time).total_seconds()
    print(f"swan total time: {dur}")
    print("swan run_time_dict max min fair measurement", run_time_dict)
    return flow_id_to_rate_mapping, dur, run_time_dict


def modify_model(model, src, dst, rate, flow_vars, constraint_dict):
    c_lb, c_ub = constraint_dict[src, dst]
    model.remove(c_lb)
    model.remove(c_ub)
    model.addConstr(flow_vars.sum(src, dst, '*') == rate)


def create_initial_mcf_model(problem:problem.Problem, link_cap, link_src_dst_path_dict, list_possible_paths,
                             frozen_flows, mcf_grb_method):
    m = Model()
    m.setParam(GRB.param.OutputFlag, 0)
    m.setParam(GRB.param.Method, mcf_grb_method)

    flow = m.addVars(list_possible_paths, lb=0, name="flow")

    m.setObjective(flow.sum('*', '*', '*'), GRB.MAXIMIZE)

    constraint_dict = dict()
    for _, (i, j, demand) in problem.sparse_commodity_list:
        if (i, j) not in frozen_flows:
            c_ub = m.addConstr(flow.sum(i, j, '*'), GRB.LESS_EQUAL, demand)
            c_lb = m.addConstr(flow.sum(i, j, '*'), GRB.GREATER_EQUAL, 0)
            constraint_dict[i, j] = (c_lb, c_ub)
        else:
            m.addConstr(flow.sum(i, j, '*') == frozen_flows[i, j])

    index = 0
    for e in link_src_dst_path_dict:
        sum_flow = 0
        for (src, dst, idx) in link_src_dst_path_dict[e]:
            sum_flow += flow[(src, dst, idx)]
        m.addConstr(sum_flow, GRB.LESS_EQUAL, link_cap, name="capacity_" + str(index))
        index += 1

    return m, flow, constraint_dict


def compute_throughput_path_based_given_tm(problem:problem.Problem, throughput_lb, throughput_ub, frozen_flows,
                                           model, constraint_dict, flow_vars, run_time_dict, break_down=False):
    if break_down:
        st_time_model = datetime.now()
    st_time = datetime.now()

    for _, (i, j, demand) in problem.sparse_commodity_list:
        if (i, j) not in frozen_flows:
            c_lb, c_ub = constraint_dict[i, j]
            c_lb.rhs = throughput_lb
            c_ub.rhs = min(demand, throughput_ub)

    if break_down:
        check_point_model = datetime.now()

    # model.reset()
    model.optimize()
    run_time_dict[MCF_SOLVER] += model.Runtime

    if break_down:
        check_point_optimize = datetime.now()

    # m.write('model.lp')
    if model.Status != GRB.OPTIMAL:
        print("not converged")
        raise Exception

    flow_vars = model.getAttr('x', flow_vars)
    flow_id_to_flow_rate_mapping = defaultdict(list)
    for (i, j, _), rate in flow_vars.items():
        flow_id_to_flow_rate_mapping[i, j].append(rate)

    run_time_dict[MCF_TOTAL] += (datetime.now() - st_time).total_seconds()
    if break_down:
        check_point_retrieve = datetime.now()
        time_model = (check_point_model - st_time_model).total_seconds()
        time_optimize = (check_point_optimize - check_point_model).total_seconds()
        time_retrieve = (check_point_retrieve - check_point_optimize).total_seconds()
        return flow_id_to_flow_rate_mapping, time_model, time_optimize, time_retrieve
    return flow_id_to_flow_rate_mapping
