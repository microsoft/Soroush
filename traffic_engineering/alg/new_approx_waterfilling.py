from collections import defaultdict
from datetime import datetime

import numpy as np
from gurobi import *

from utilities import constants
from ncflow.lib import problem


def max_min_approx(alpha, U, problem:problem.Problem, paths, link_cap, max_epsilon,
                   break_down=False, link_cap_scale_multiplier=1):
    st_time = datetime.now()

    list_possible_paths = tuplelist()
    link_src_dst_path_dict = tupledict()
    list_flows = tuplelist()
    index = 0
    frozen_flows = dict()

    max_demand = np.max(problem.traffic_matrix.tm)
    T = max(1, int(np.ceil(np.log(max_demand / U) / np.log(alpha))))
    bounds = U * np.power(alpha, np.arange(T + 1)) / link_cap_scale_multiplier
    epsilon = np.power(max_epsilon, 1.0 / T)
    multiply_coeffs = np.power(epsilon, np.arange(T))
    link_cap /= link_cap_scale_multiplier

    for _, (src, dst, demand) in problem.sparse_commodity_list:
        if src == dst:
            continue

        for path in paths[(src, dst)]:
            list_possible_paths.append((src, dst, index))
            for i in range(1, len(path)):
                if (path[i - 1], path[i]) not in link_src_dst_path_dict:
                    link_src_dst_path_dict[path[i - 1], path[i]] = list()
                link_src_dst_path_dict[path[i - 1], path[i]].append((src, dst, index))
            index += 1
        if demand < U:
            frozen_flows[src, dst] = demand / link_cap_scale_multiplier
            list_flows.append((src, dst, 0))
        else:
            list_flows.append((src, dst, 0))
            for k in range(T):
                if U * np.power(alpha, k) > demand:
                    break
                list_flows.append((src, dst, k + 1))

    m = Model()
    m.setParam(GRB.param.OutputFlag, 0)
    m.setParam(GRB.param.Method, 2)

    flowpath = m.addVars(list_possible_paths, lb=0, name="flowpath")
    flow = m.addVars(list_flows, lb=0, name="flow")

    obj = 0
    init_rate = U / link_cap_scale_multiplier
    for _, (i, j, demand) in problem.sparse_commodity_list:
        m.addLConstr(flow.sum(i, j, '*') == flowpath.sum(i, j, '*'))
        if (i, j) not in frozen_flows:
            m.addLConstr(flow.sum(i, j, '*'), GRB.LESS_EQUAL, demand / link_cap_scale_multiplier)
            m.addLConstr(flow.sum(i, j, '*'), GRB.GREATER_EQUAL, 0)
            m.addLConstr(flow[i, j, 0] == init_rate)
            for k in range(T):
                if U * np.power(alpha, k) > demand:
                    break
                m.addLConstr(flow[i, j, k + 1], GRB.LESS_EQUAL, bounds[k + 1] - bounds[k])
                obj += multiply_coeffs[k] * flow[i, j, k + 1]
        else:
            m.addLConstr(flow.sum(i, j, '*') == frozen_flows[i, j])

    # for _, (i, j, demand) in problem.sparse_commodity_list:
    #     if (i, j) not in frozen_flows:
    #         m.addConstr(flow.sum(i, j, '*', '*'), GRB.LESS_EQUAL, demand / link_cap_scale_multiplier)
    #         m.addConstr(flow.sum(i, j, '*', '*'), GRB.GREATER_EQUAL, 0)
    #         m.addConstr(flow.sum(i, j, '*', 0) == U / link_cap_scale_multiplier)
    #         for k in range(T):
    #             if U * np.power(alpha, k) > demand:
    #                 continue
    #             m.addConstr(flow.sum(i, j, '*', k + 1), GRB.LESS_EQUAL, bounds[k + 1] - bounds[k])
    #             obj += multiply_coeffs[k] * flow.sum(i, j, '*', k + 1)
    #     else:
    #         m.addConstr(flow.sum(i, j, '*', '*') == frozen_flows[i, j])

    for e in link_src_dst_path_dict:
        sum_flow = quicksum(flowpath[key] for key in link_src_dst_path_dict[e])
        m.addLConstr(sum_flow, GRB.LESS_EQUAL, link_cap, name=f"capacity_{e}")

    m.setObjective(obj, GRB.MAXIMIZE)

    if break_down:
        check_point_model = datetime.now()

    m.optimize()

    if break_down:
        check_point_optimize = datetime.now()

    if m.Status != GRB.OPTIMAL:
        print("not converged")

    flow_vars = m.getAttr('x', flow)
    flow_id_to_flow_rate_mapping = defaultdict(list)
    for (src, dst, _), rate in flow_vars.items():
        flow_id_to_flow_rate_mapping[(src, dst)].append(rate * link_cap_scale_multiplier)

    if break_down:
        check_point_retrieve = datetime.now()
        time_model = (check_point_model - st_time).total_seconds()
        time_optimize = (check_point_optimize - check_point_model).total_seconds()
        time_retrieve = (check_point_retrieve - check_point_optimize).total_seconds()
        return flow_id_to_flow_rate_mapping, (time_model, time_optimize, time_retrieve)

    duration = (datetime.now() - st_time).total_seconds()
    return flow_id_to_flow_rate_mapping, duration
