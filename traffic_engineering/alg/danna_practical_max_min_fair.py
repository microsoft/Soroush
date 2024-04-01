from collections import defaultdict
from datetime import datetime

import numpy as np
from gurobi import *

from utilities import constants
from ncflow.lib import problem


def get_rates(U, problem:problem.Problem, paths, link_cap, break_down=False):
    st_time = datetime.now()
    max_demand = 0
    min_demand = np.inf
    src_dst_demand_mapping = dict()

    list_possible_paths = tuplelist()
    link_src_dst_path_dict = tupledict()
    index = 0
    frozen_flows = dict()
    unfrozen_flows = dict()

    list_demands = [0.0]

    for fid, (src, dst, demand) in problem.sparse_commodity_list:
        max_demand = max(max_demand, demand)
        min_demand = min(min_demand, demand)
        src_dst_demand_mapping[src, dst] = demand
        list_demands.append(demand)
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
            frozen_flows[src, dst] = demand
        else:
            unfrozen_flows[src, dst] = demand

    num_flows = len(problem.sparse_commodity_list)
    sorted_list_unique_demands = sorted(list(set(list_demands)))
    prev_id_feas = 0
    iter_no = 0
    feas_model, feas_constraint_dict = create_initial_feasibility_model(problem, link_cap, link_src_dst_path_dict, list_possible_paths)
    mcf_model, mcf_thru_var, mcf_constraint_dict, flow_var = create_initial_mcf_model(problem, link_cap, link_src_dst_path_dict, list_possible_paths)

    if break_down:
        pre_dur = (datetime.now() - st_time).total_seconds()
        exponential_search_dur = 0
        binary_search_dur = 0
        freeze_demand_limited_dur = 0
        create_model_freeze_link_dur = 0
        solve_optimization_freeze_link_dur = 0
        retrieve_values_freeze_link_dur = 0

    while len(frozen_flows) < num_flows:
        # if iter_no == 49:
        #     print("found")
        print(f"iter no. {iter_no}, num remaining flows {len(unfrozen_flows)}, total flows {num_flows}")
        prev_id_feas = binary_search_saturated_demands(problem, frozen_flows, unfrozen_flows, feas_model, feas_constraint_dict,
                                                       prev_id_feas, sorted_list_unique_demands, break_down=break_down)

        if break_down:
            prev_id_feas, e_d, b_d, f_d = prev_id_feas
            exponential_search_dur += e_d
            binary_search_dur += b_d
            freeze_demand_limited_dur += f_d

        if len(frozen_flows) == num_flows:
            break
        output = find_next_saturated_edge(problem, frozen_flows, unfrozen_flows, mcf_model, mcf_thru_var,
                                          mcf_constraint_dict, link_src_dst_path_dict, flow_var, link_cap,
                                          debug=False, break_down=break_down)
        if break_down:
            time_model, time_optimize, time_retrieve = output
            create_model_freeze_link_dur += time_model
            solve_optimization_freeze_link_dur += time_optimize
            retrieve_values_freeze_link_dur += time_retrieve
        iter_no += 1

    if break_down:
        dur = (pre_dur, exponential_search_dur, binary_search_dur, freeze_demand_limited_dur,
               create_model_freeze_link_dur, solve_optimization_freeze_link_dur, retrieve_values_freeze_link_dur)
    else:
        dur = (datetime.now() - st_time).total_seconds()
    print("danna practical max min fair dur: ", dur)
    return frozen_flows, dur


def binary_search_saturated_demands(problem:problem.Problem, frozen_flows, unfrozen_flows, model, constraint_dict,
                                    prev_id_feas, sorted_list_unique_demands, break_down=False):

    if break_down:
        checkpoint1 = datetime.now()

    id_feasible = prev_id_feas
    post_exp_search_id_feas = prev_id_feas
    diff = 1
    len_unique_demands = len(sorted_list_unique_demands)
    found_infeasible = False
    while not found_infeasible:
        id_infeasible = np.minimum(id_feasible + diff, len_unique_demands - 1)
        throughput_lb = sorted_list_unique_demands[id_infeasible]
        found_feasible = perform_feasibility_test(problem, throughput_lb, frozen_flows, model, constraint_dict)
        if found_feasible:
            post_exp_search_id_feas = id_infeasible
        found_infeasible = not found_feasible
        if id_infeasible >= len_unique_demands - 1:
            break
        diff *= 2

    id_feasible = post_exp_search_id_feas

    if id_infeasible >= len_unique_demands - 1 and not found_infeasible:
        for (src, dst) in unfrozen_flows:
            frozen_flows[src, dst] = unfrozen_flows[src, dst]
        unfrozen_flows.clear()
        if break_down:
            time_dur = (datetime.now() - checkpoint1).total_seconds()
            return len_unique_demands, time_dur, 0, 0
        return len_unique_demands

    if break_down:
        checkpoint2 = datetime.now()

    id_current = int(np.ceil((id_feasible + id_infeasible) / 2))
    while id_infeasible - id_feasible > 1:
        t_current = sorted_list_unique_demands[id_current]
        feasible = perform_feasibility_test(problem, t_current, frozen_flows, model, constraint_dict)
        if feasible:
            id_feasible = id_current
        else:
            id_infeasible = id_current
        id_current = int(np.ceil((id_feasible + id_infeasible) / 2))

    if break_down:
        checkpoint3 = datetime.now()

    set_to_remove = set()
    if sorted_list_unique_demands[id_feasible] > 0:
        for (src, dst) in unfrozen_flows:
            if unfrozen_flows[src, dst] <= sorted_list_unique_demands[id_feasible]:
                frozen_flows[src, dst] = np.minimum(sorted_list_unique_demands[id_feasible], unfrozen_flows[src, dst])
                set_to_remove.add((src, dst))

    for (src, dst) in set_to_remove:
        del unfrozen_flows[src, dst]

    if break_down:
        checkpoint4 = datetime.now()
        exponential_search_dur = (checkpoint2 - checkpoint1).total_seconds()
        binary_search_dur = (checkpoint3 - checkpoint2).total_seconds()
        freezing_demand_limited_dur = (checkpoint4 - checkpoint3).total_seconds()
        return id_feasible, exponential_search_dur, binary_search_dur, freezing_demand_limited_dur
    return id_feasible


def create_initial_feasibility_model(problem:problem.Problem, link_cap, link_src_dst_path_dict, list_possible_paths):
    m = Model()
    m.setParam(GRB.param.OutputFlag, 0)
    # m.setParam(GRB.param.Method, 2) # method 2 does not necessarily win and concurrent method is much faster
    flow = m.addVars(list_possible_paths, lb=0, name="flow")

    constraint_dict = dict()
    for _, (i, j, demand) in problem.sparse_commodity_list:
        c_ub = m.addConstr(flow.sum(i, j, '*'), GRB.LESS_EQUAL, demand)
        c_lb = m.addConstr(flow.sum(i, j, '*'), GRB.GREATER_EQUAL, 0)
        constraint_dict[i, j] = (c_lb, c_ub)

    index = 0
    for e in link_src_dst_path_dict:
        sum_flow = 0
        for (src, dst, idx) in link_src_dst_path_dict[e]:
            sum_flow += flow[(src, dst, idx)]
        m.addConstr(sum_flow, GRB.LESS_EQUAL, link_cap, name="capacity_" + str(index))
        index += 1

    return m, constraint_dict


def perform_feasibility_test(problem:problem.Problem, throughput_lb, frozen_flows, model, constraint_dict):

    for _, (i, j, demand) in problem.sparse_commodity_list:
        c_lb, c_ub = constraint_dict[i, j]
        if (i, j) not in frozen_flows:
            c_lb.rhs = min(demand, throughput_lb)
        else:
            c_lb.rhs = frozen_flows[i, j]

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return False
    return True


def create_initial_mcf_model(problem:problem.Problem, link_cap, link_src_dst_path_dict, list_possible_paths):
    m = Model()
    m.setParam(GRB.param.OutputFlag, 0)
    # m.setParam(GRB.param.Method, 2) # method 2 does not necessarily win and concurrent method is much faster
    flow = m.addVars(list_possible_paths, lb=0, name="flow")

    t = m.addVar(lb=0, name="throughput")
    m.setObjective(t, GRB.MAXIMIZE)

    constraint_dict = dict()
    for _, (i, j, demand) in problem.sparse_commodity_list:
        c_ub = m.addConstr(flow.sum(i, j, '*'), GRB.LESS_EQUAL, demand)
        c_lb = m.addConstr(flow.sum(i, j, '*'), GRB.GREATER_EQUAL, t)
        constraint_dict[i, j] = (c_lb, c_ub)

    index = 0
    for e in link_src_dst_path_dict:
        sum_flow = 0
        for (src, dst, idx) in link_src_dst_path_dict[e]:
            sum_flow += flow[(src, dst, idx)]
        m.addConstr(sum_flow, GRB.LESS_EQUAL, link_cap, name="capacity_" + str(index))
        index += 1
    return m, t, constraint_dict, flow


def find_next_saturated_edge(problem:problem.Problem, frozen_flows, unfrozen_flows, model, throughput_var,
                             constraint_dict, link_src_dst_path_dict, flow_var, link_cap, break_down=False, debug=False):
    if break_down:
        st_time_model = datetime.now()

    unfrozen_constraint_list = []
    for _, (i, j, demand) in problem.sparse_commodity_list:
        c_lb, c_ub = constraint_dict[i, j]
        if (i, j) not in frozen_flows:
            unfrozen_constraint_list.append((c_lb, i, j))
        else:
            c_lb.rhs = frozen_flows[i, j]
            model.chgCoeff(c_lb, throughput_var, 0)

    if break_down:
        check_point_model = datetime.now()

    model.optimize()

    if break_down:
        check_point_optimize = datetime.now()

    if model.Status != GRB.OPTIMAL:
        print("not converged")
        raise Exception("optimization not converged")

    throughput = throughput_var.X
    for constr, src, dst in unfrozen_constraint_list:
        dual_val = constr.getAttr(GRB.attr.Pi)
        if dual_val < 0:
            frozen_flows[src, dst] = throughput
            del unfrozen_flows[src, dst]

    if debug:
        print("================= debugging link capacities =================")
        flow_thru = model.getAttr('x', flow_var)
        link_util = defaultdict(int)
        for e in link_src_dst_path_dict:
            for (src, dst, idx) in link_src_dst_path_dict[e]:
                if (src, dst) in frozen_flows:
                    link_util[e] += flow_thru[src, dst, idx]
        for e in link_util:
            if link_util[e] > link_cap + constants.O_epsilon:
                print(f"capacity violation: {e} {link_util[e]}/{link_cap}")
                raise Exception("capacity violated")

    if break_down:
        check_point_retrieve = datetime.now()
        time_model = (check_point_model - st_time_model).total_seconds()
        time_optimize = (check_point_optimize - check_point_model).total_seconds()
        time_retrieve = (check_point_retrieve - check_point_optimize).total_seconds()
        return time_model, time_optimize, time_retrieve
    return None