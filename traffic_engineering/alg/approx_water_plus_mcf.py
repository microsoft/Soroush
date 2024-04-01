from collections import defaultdict
from datetime import datetime

import numpy as np
import cvxpy as cp
from gurobi import *

from alg import approx_waterfilling
from ncflow.lib import problem
from utilities import constants


def get_rates(problem: problem.Problem, paths, path_edge_idx_mapping, num_paths_per_flow, num_iter, link_cap, break_down=False):
    st_time = datetime.now()
    flow_details = problem.sparse_commodity_list
    approx_waterfilling_fid_to_flow_rate, dur = approx_waterfilling.get_rates(problem, path_edge_idx_mapping,
                                                                              num_paths_per_flow, num_iter, link_cap,
                                                                              break_down)

    list_possible_paths = tuplelist()
    link_src_dst_path_dict = tupledict()
    index = 0
    for _, (src, dst, demand) in flow_details:
        if src == dst:
            continue
        for path in paths[(src, dst)]:
            list_possible_paths.append((src, dst, index))
            for i in range(1, len(path)):
                if (path[i - 1], path[i]) not in link_src_dst_path_dict:
                    link_src_dst_path_dict[path[i - 1], path[i]] = list()
                link_src_dst_path_dict[path[i - 1], path[i]].append((src, dst, index))
            index += 1

    output = compute_throughput_path_based_given_tm(flow_details, approx_waterfilling_fid_to_flow_rate, link_cap,
                                                    link_src_dst_path_dict, list_possible_paths, break_down=break_down)

    dur = (datetime.now() - st_time).total_seconds()
    return output, dur


# def compute_throughput_path_based_given_tm(flow_details, flow_rate_matrix, link_cap, num_paths_per_flow,
#                                            demand_vector, fid_edge_vector, path_existence_vector, epsilon, k, alpha,
#                                            beta, break_down=False):

########################## version 1
# def compute_throughput_path_based_given_tm(flow_details, fid_to_throughput_lb, link_cap,
#                                            link_src_dst_path_dict, list_possible_paths, epsilon, alpha, beta, k,
#                                            break_down=False):
#     if break_down:
#         st_time_model = datetime.now()
#
#     num_flows = len(flow_details)
#     throughput_per_flow = np.add.reduce(fid_to_throughput_lb, axis=1)
#     sorted_fids = np.argsort(throughput_per_flow)
#     obj_coeff_vector = np.power(epsilon, np.arange(num_flows))
#     multi_coeff_flows = np.empty(num_flows)
#     multi_coeff_flows[sorted_fids] = obj_coeff_vector
#     additive_term = k * np.power(beta, np.arange(num_flows - 1, 0, -1))
#     multiplicative_term = multi_coeff_flows * (1 - alpha) + alpha
#
#     m = Model()
#     m.setParam(GRB.param.OutputFlag, 1)
#     m.setParam(GRB.param.Method, 2)
#     m.setParam(GRB.param.Crossover, 0)
#     flow = m.addVars(list_possible_paths, lb=0, name="flow")
#
#     # m.setObjective(flow.sum('*', '*', '*'), GRB.MAXIMIZE)
#     obj = 0
#     for fid, (i, j, demand) in flow_details:
#         total_flow = flow.sum(fid, '*')
#         obj += multiplicative_term[fid] * total_flow
#         m.addConstr(total_flow, GRB.LESS_EQUAL, demand)
#     m.setObjective(obj, GRB.MAXIMIZE)
#
#     for idx in range(num_flows - 1):
#         st_fid = sorted_fids[idx]
#         en_fid = sorted_fids[idx + 1]
#
#         st_total_flow = flow.sum(st_fid, '*')
#         en_total_flow = flow.sum(en_fid, '*')
#         m.addConstr(st_total_flow - en_total_flow, GRB.LESS_EQUAL, additive_term[idx])
#         # total_rate = np.sum(fid_to_throughput_lb[fid])
#         # if demand > total_rate + constants.O_epsilon:
#         #     m.addConstr(flow.sum(i, j, '*'), GRB.LESS_EQUAL, demand)
#         #     m.addConstr(flow.sum(i, j, '*'), GRB.GREATER_EQUAL, total_rate)
#         # else:
#         #     m.addConstr(flow.sum(i, j, '*') == demand)
#
#     index = 0
#     for e in link_src_dst_path_dict:
#         sum_flow = 0
#         for (fid, pid) in link_src_dst_path_dict[e]:
#             sum_flow += flow[(fid, pid)]
#         m.addConstr(sum_flow, GRB.LESS_EQUAL, link_cap, name="capacity_" + str(index))
#         index += 1
#
#     # for (fid, p_idx) in list_possible_paths:
#     #     flow[(fid, p_idx)].Start = fid_to_throughput_lb[fid, p_idx]
#
#
#     if break_down:
#         check_point_model = datetime.now()
#
#     m.optimize()
#
#     if break_down:
#         check_point_optimize = datetime.now()
#
#     # m.write('model.lp')
#     if m.Status != GRB.OPTIMAL:
#         print("not converged")
#
#     flow_vars = m.getAttr('x', flow)
#     flow_id_to_flow_rate_mapping = defaultdict(list)
#     for (fid, _), rate in flow_vars.items():
#         flow_id_to_flow_rate_mapping[fid].append(rate)
#
#     if break_down:
#         check_point_retrieve = datetime.now()
#         time_model = (check_point_model - st_time_model).total_seconds()
#         time_optimize = (check_point_optimize - check_point_model).total_seconds()
#         time_retrieve = (check_point_retrieve - check_point_optimize).total_seconds()
#         return flow_id_to_flow_rate_mapping, (time_model, time_optimize, time_retrieve)
#     return flow_id_to_flow_rate_mapping

################ Version 2
# def compute_throughput_path_based_given_tm(flow_details, flow_rate_matrix, link_cap, num_paths_per_flow,
#                                            demand_vector, fid_edge_vector, path_existence_vector, epsilon, k, alpha,
#                                            beta, break_down=False):
#     if break_down:
#         st_time_model = datetime.now()
#
#     num_flows = len(flow_details)
#     rate_vars = cp.Variable((num_flows, num_paths_per_flow))
#
#     throughput_per_flow = np.add.reduce(flow_rate_matrix, axis=1)
#     sorted_fids = np.argsort(throughput_per_flow)
#     obj_coeff_vector = np.power(epsilon, np.arange(num_flows))
#     multi_coeff_flows = np.empty(num_flows)
#     multi_coeff_flows[sorted_fids] = obj_coeff_vector
#
#     additive_term = k * np.power(beta, np.arange(num_flows - 1, 0, -1))
#     multiplicative_term = multi_coeff_flows * (1 - alpha) + alpha
#     eff_thru_per_flow = cp.sum(rate_vars, axis=1)
#     # objective = cp.Maximize(cp.sum(cp.multiply(multiplicative_term, eff_thru_per_flow)))
#     objective = cp.Maximize(cp.sum(eff_thru_per_flow))
#
#     exist_less_path = (not np.logical_and.reduce(path_existence_vector))
#
#     constraints = [rate_vars >= 0,
#                    cp.matmul(fid_edge_vector.T, cp.vec(rate_vars)) <= link_cap,
#                    eff_thru_per_flow <= demand_vector,
#                    eff_thru_per_flow >= np.sum(flow_rate_matrix, axis=1)]
#                    # eff_thru_per_flow[sorted_fids[:-1]] - eff_thru_per_flow[sorted_fids[1:]] <= additive_term]
#     if exist_less_path:
#         non_existence_vars = cp.vec(rate_vars)[1 - path_existence_vector]
#         constraints.append(non_existence_vars == 0)
#
#     model = cp.Problem(objective, constraints)
#     if break_down:
#         checkpoint1 = datetime.now()
#     result = model.solve(solver=cp.GUROBI, verbose=True)
#     if break_down:
#         checkpoint2 = datetime.now()
#         time_model = (checkpoint1 - st_time_model).total_seconds()
#         time_solve = (checkpoint2 - checkpoint1).total_seconds()
#         return rate_vars.value, (time_model, time_solve)
#     return rate_vars.value
#

################### Verson 3
def compute_throughput_path_based_given_tm(flow_details, fid_to_throughput_lb, link_cap, demand_vector,
                                           link_src_dst_path_dict, list_possible_paths, epsilon, alpha, beta, k,
                                           num_flows_per_barrier, break_down=False, link_cap_scale_multiplier=1):
    if break_down:
        st_time_model = datetime.now()

    num_flows = len(flow_details)
    throughput_per_flow = np.add.reduce(fid_to_throughput_lb, axis=1)
    sorted_fids = np.argsort(throughput_per_flow)
    # obj_coeff_vector = np.power(epsilon, np.arange(num_flows))
    # multi_coeff_flows = np.empty(num_flows)
    # multi_coeff_flows[sorted_fids] = obj_coeff_vector
    num_bins = np.int(np.ceil(num_flows / num_flows_per_barrier))
    additive_term = k * np.power(beta, np.arange(num_bins - 1, 0, -1)) / link_cap_scale_multiplier
    # print(additive_term[:num_bins - 1])
    multi_coeff_flows = np.power(epsilon, np.arange(num_bins))
    multiplicative_term = multi_coeff_flows * (1 - alpha) + alpha
    demand_vector /= link_cap_scale_multiplier
    link_cap /= link_cap_scale_multiplier
    # print("0", (datetime.now() - st_time_model).total_seconds())

    m = Model()
    m.setParam(GRB.param.OutputFlag, 0)
    m.setParam(GRB.param.Method, 2)
    # m.setParam(GRB.param.Crossover, 0)
    # m.setParam(GRB.param.NumericFocus, 1)
    # m.setParam(GRB.param.Presolve, 0)
    # m.setParam(GRB.param.LPWarmStart, 1)
    flow = m.addVars(list_possible_paths, lb=0, name="flow")

    obj = 0
    # print("1", (datetime.now() - st_time_model).total_seconds())
    t_lb = m.addVars(range(num_bins - 1), lb=0, name=f"t_lb_0")
    idx = 0
    for bin_idx in range(num_bins):
        flow_coefficient = multiplicative_term[bin_idx]
        for i in range(num_flows_per_barrier):
            fid = sorted_fids[idx]
            total_flow = flow.sum(fid, '*')
            obj += flow_coefficient * total_flow
            m.addConstr(total_flow, GRB.LESS_EQUAL, demand_vector[fid])
            if bin_idx > 0:
                m.addConstr(total_flow, GRB.GREATER_EQUAL, t_lb[bin_idx - 1])
            if bin_idx < num_bins - 1:
                m.addConstr(total_flow, GRB.LESS_EQUAL, t_lb[bin_idx] + additive_term[bin_idx])
            idx += 1
            if idx >= num_flows:
                break
        bin_idx += 1
        if idx >= num_flows:
            break

    # print("2", (datetime.now() - st_time_model).total_seconds())

    m.setObjective(obj, GRB.MAXIMIZE)
    # st_fid = sorted_fids[idx]
    # en_fid = sorted_fids[idx + 1]
    #
    # st_total_flow = flow.sum(st_fid, '*')
    # en_total_flow = flow.sum(en_fid, '*')
    # m.addConstr(st_total_flow - en_total_flow, GRB.LESS_EQUAL, additive_term[idx])
    # total_rate = np.sum(fid_to_throughput_lb[fid])
    # if demand > total_rate + constants.O_epsilon:
    #     m.addConstr(flow.sum(i, j, '*'), GRB.LESS_EQUAL, demand)
    #     m.addConstr(flow.sum(i, j, '*'), GRB.GREATER_EQUAL, total_rate)
    # else:
    #     m.addConstr(flow.sum(i, j, '*') == demand)

    for e in link_src_dst_path_dict:
        sum_flow = quicksum(flow[key] for key in link_src_dst_path_dict[e])
        m.addConstr(sum_flow, GRB.LESS_EQUAL, link_cap, name=f"capacity_{e}")

    # m.update()
    # print("3", (datetime.now() - st_time_model).total_seconds())
    # for (fid, pid) in list_possible_paths:
    #     flow[(fid, pid)].PStart = fid_to_throughput_lb[fid, pid]
    #
    # for i in range(num_bins - 1):
    #     t_lb[i].PStart = init_sol_t_lb[i]

    if break_down:
        check_point_model = datetime.now()

    m.optimize()

    if break_down:
        check_point_optimize = datetime.now()

    # m.write('model.lp')
    if m.Status != GRB.OPTIMAL:
        print("not converged")

    flow_vars = m.getAttr('x', flow)
    # t_vars = m.getAttr('x', t_lb)
    # print(t_vars)
    flow_id_to_flow_rate_mapping = defaultdict(list)
    for (fid, _), rate in flow_vars.items():
        flow_id_to_flow_rate_mapping[fid].append(rate * link_cap_scale_multiplier)

    if break_down:
        check_point_retrieve = datetime.now()
        time_model = (check_point_model - st_time_model).total_seconds()
        time_optimize = (check_point_optimize - check_point_model).total_seconds()
        time_retrieve = (check_point_retrieve - check_point_optimize).total_seconds()
        return flow_id_to_flow_rate_mapping, (time_model, time_optimize, time_retrieve)
    return flow_id_to_flow_rate_mapping

