from collections import defaultdict
from datetime import datetime

import numpy as np
from gurobi import *

from utilities import constants
from ncflow.lib import problem

MODEL_TIME = 'model'
FEASIBILITY_TOTAL = 'feasibility_total'
FEASIBILITY_SOLVER = 'feasibility_solver'
MCF_TOTAL = 'mcf_total'
MCF_SOLVER = 'mcf_solver'
EXTRACT_RATE = 'extract_rate'


def get_rates(U, problem: problem.Problem, paths, link_cap, feasibility_grb_method=1, mcf_grb_method=2,
              break_down=False, numeric_num_decimals=-1):
    run_time_dict = dict()
    run_time_dict[MODEL_TIME] = 0
    run_time_dict[FEASIBILITY_TOTAL] = 0
    run_time_dict[FEASIBILITY_SOLVER] = 0
    run_time_dict[MCF_TOTAL] = 0
    run_time_dict[MCF_SOLVER] = 0
    run_time_dict[EXTRACT_RATE] = 0
    st_time = datetime.now()
    max_demand = 0
    min_demand = np.inf
    src_dst_demand_mapping = dict()

    list_possible_paths = tuplelist()
    link_src_dst_path_dict = tupledict()
    frozen_flows = dict()
    unfrozen_flows = dict()

    list_demands = [0.0]
    flow_id_to_flow_rate_mapping = defaultdict(list)
    for fid, (src, dst, demand) in problem.sparse_commodity_list:
        max_demand = max(max_demand, demand)
        min_demand = min(min_demand, demand)
        src_dst_demand_mapping[src, dst] = demand
        list_demands.append(demand)
        if src == dst:
            continue
        index = 0
        for path in paths[(src, dst)]:
            list_possible_paths.append((src, dst, index))
            for i in range(1, len(path)):
                if (path[i - 1], path[i]) not in link_src_dst_path_dict:
                    link_src_dst_path_dict[path[i - 1], path[i]] = list()
                link_src_dst_path_dict[path[i - 1], path[i]].append((src, dst, index))
            flow_id_to_flow_rate_mapping[src, dst].append(0)
            index += 1
        if demand < U:
            frozen_flows[src, dst] = demand
        else:
            unfrozen_flows[src, dst] = demand

    num_flows = len(problem.sparse_commodity_list)
    sorted_list_unique_demands = sorted(list(set(list_demands)))
    prev_id_feas = 0
    iter_no = 0

    cf_output = create_initial_feasibility_model(problem, link_cap, link_src_dst_path_dict, list_possible_paths, feasibility_grb_method)
    feas_model, feas_constraint_dict = cf_output

    cmcf_output = create_initial_mcf_model(problem, link_cap, link_src_dst_path_dict, list_possible_paths, mcf_grb_method)
    mcf_model, mcf_thru_var, mcf_constraint_dict, flow_var = cmcf_output

    if break_down:
        pre_dur = (datetime.now() - st_time).total_seconds()
        exponential_search_dur = 0
        binary_search_dur = 0
        freeze_demand_limited_dur = 0
        create_model_freeze_link_dur = 0
        solve_optimization_freeze_link_dur = 0
        retrieve_values_freeze_link_dur = 0
    run_time_dict[MODEL_TIME] = (datetime.now() - st_time).total_seconds()

    last_model = None
    while len(frozen_flows) < num_flows:
        print(f"iter no. {iter_no}, num remaining flows {len(unfrozen_flows)}, total flows {num_flows}")
        prev_id_feas = binary_search_saturated_demands(problem, frozen_flows, unfrozen_flows, feas_model, feas_constraint_dict,
                                                       prev_id_feas, sorted_list_unique_demands, run_time_dict, break_down=break_down)
        last_model = feas_model

        if break_down:
            prev_id_feas, e_d, b_d, f_d = prev_id_feas
            exponential_search_dur += e_d
            binary_search_dur += b_d
            freeze_demand_limited_dur += f_d

        if len(frozen_flows) == num_flows:
            break
        output = find_next_saturated_edge(problem, frozen_flows, unfrozen_flows, mcf_model, mcf_thru_var,
                                          mcf_constraint_dict, link_src_dst_path_dict, flow_var, link_cap,
                                          run_time_dict, debug=False, break_down=break_down, numeric_num_decimals=numeric_num_decimals)
        last_model = mcf_model
        if break_down:
            time_model, time_optimize, time_retrieve = output
            create_model_freeze_link_dur += time_model
            solve_optimization_freeze_link_dur += time_optimize
            retrieve_values_freeze_link_dur += time_retrieve
        iter_no += 1

    ex_st_time = datetime.now()
    if last_model is None:
        for (i, j), rate in frozen_flows.items():
            num_paths = len(flow_id_to_flow_rate_mapping[i, j])
            uniform_rate = rate / num_paths
            for idx in range(num_paths):
                flow_id_to_flow_rate_mapping[i, j][idx] = uniform_rate
    else:
        flow_vars = last_model.getAttr('x', flow_var)
        for (i, j, pidx), rate in flow_vars.items():
            if (i, j) in frozen_flows:
                flow_id_to_flow_rate_mapping[i, j][pidx] = rate
    run_time_dict[EXTRACT_RATE] = (datetime.now() - ex_st_time).total_seconds()

    if break_down:
        dur = (pre_dur, exponential_search_dur, binary_search_dur, freeze_demand_limited_dur,
               create_model_freeze_link_dur, solve_optimization_freeze_link_dur, retrieve_values_freeze_link_dur)
    else:
        dur = (datetime.now() - st_time).total_seconds()
    print("danna practical max min fair dur: ", dur)
    print("danna run_time_dict max min fair measurement", run_time_dict)
    return frozen_flows, flow_id_to_flow_rate_mapping, dur, run_time_dict


def binary_search_saturated_demands(problem: problem.Problem, frozen_flows, unfrozen_flows, model, constraint_dict,
                                    prev_id_feas, sorted_list_unique_demands, run_time_dict, break_down=False):

    if break_down:
        checkpoint1 = datetime.now()
    st_time = datetime.now()

    id_feasible = prev_id_feas
    post_exp_search_id_feas = prev_id_feas
    diff = 1
    len_unique_demands = len(sorted_list_unique_demands)
    found_infeasible = False
    while not found_infeasible:
        id_infeasible = np.minimum(id_feasible + diff, len_unique_demands - 1)
        throughput_lb = sorted_list_unique_demands[id_infeasible]
        found_feasible = perform_feasibility_test(problem, throughput_lb, frozen_flows, model, constraint_dict,
                                                  run_time_dict)
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
        run_time_dict[FEASIBILITY_TOTAL] += (datetime.now() - st_time).total_seconds()
        if break_down:
            time_dur = (datetime.now() - checkpoint1).total_seconds()
            return len_unique_demands, time_dur, 0, 0
        return len_unique_demands

    if break_down:
        checkpoint2 = datetime.now()

    id_current = int(np.ceil((id_feasible + id_infeasible) / 2))
    while id_infeasible - id_feasible > 1:
        t_current = sorted_list_unique_demands[id_current]
        feasible = perform_feasibility_test(problem, t_current, frozen_flows, model, constraint_dict,
                                            run_time_dict)
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

    run_time_dict[FEASIBILITY_TOTAL] += (datetime.now() - st_time).total_seconds()

    if break_down:
        checkpoint4 = datetime.now()
        exponential_search_dur = (checkpoint2 - checkpoint1).total_seconds()
        binary_search_dur = (checkpoint3 - checkpoint2).total_seconds()
        freezing_demand_limited_dur = (checkpoint4 - checkpoint3).total_seconds()
        return id_feasible, exponential_search_dur, binary_search_dur, freezing_demand_limited_dur
    return id_feasible


def create_initial_feasibility_model(problem: problem.Problem, link_cap, link_src_dst_path_dict, list_possible_paths,
                                     feasibility_grb_method):
    m = Model()
    m.setParam(GRB.param.OutputFlag, 0)
    m.setParam(GRB.param.Method, feasibility_grb_method)
    m.setParam(GRB.param.NumericFocus, 3)
    m.setParam(GRB.param.FeasibilityTol, 0.000000001)
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
        m.addConstr(sum_flow, GRB.LESS_EQUAL, link_cap)
        index += 1

    return m, constraint_dict


def perform_feasibility_test(problem: problem.Problem, throughput_lb, frozen_flows, model, constraint_dict,
                             run_time_dict):

    for _, (i, j, demand) in problem.sparse_commodity_list:
        c_lb, c_ub = constraint_dict[i, j]
        if (i, j) not in frozen_flows:
            c_lb.rhs = min(demand, throughput_lb)
        else:
            c_lb.rhs = frozen_flows[i, j]

    model.optimize()
    run_time_dict[FEASIBILITY_SOLVER] += model.Runtime

    if model.Status != GRB.OPTIMAL:
        return False
    return True


def create_initial_mcf_model(problem: problem.Problem, link_cap, link_src_dst_path_dict, list_possible_paths, mcf_grb_method):
    m = Model()
    m.setParam(GRB.param.OutputFlag, 0)
    m.setParam(GRB.param.Method, mcf_grb_method)
    m.setParam(GRB.param.NumericFocus, 3)
    m.setParam(GRB.param.FeasibilityTol, 0.000000001)
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
        m.addConstr(sum_flow, GRB.LESS_EQUAL, link_cap)
        index += 1
    return m, t, constraint_dict, flow


def find_next_saturated_edge(problem: problem.Problem, frozen_flows, unfrozen_flows, model, throughput_var,
                             constraint_dict, link_src_dst_path_dict, flow_var, link_cap,
                             run_time_dict, break_down=False, debug=False, numeric_num_decimals=-1):
    if break_down:
        st_time_model = datetime.now()
    st_time = datetime.now()

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
    run_time_dict[MCF_SOLVER] += model.Runtime

    if break_down:
        check_point_optimize = datetime.now()

    if model.Status != GRB.OPTIMAL:
        print("not converged")
        raise Exception("optimization not converged")

    throughput = throughput_var.X
    for constr, src, dst in unfrozen_constraint_list:
        dual_val = constr.getAttr(GRB.attr.Pi)
        if dual_val < 0:
            if numeric_num_decimals <= 0:
                set_throughput = throughput
            else:
                decimal_factor = np.power(10, numeric_num_decimals)
                set_throughput = np.floor(throughput * decimal_factor) / decimal_factor
                set_throughput = np.minimum(unfrozen_flows[src, dst], set_throughput)

            # print(f"set demand between {src} and {dst} to {set_throughput}")
            frozen_flows[src, dst] = set_throughput
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
            # print(f"used capacity {e}: {link_util[e]}")
            if link_util[e] > link_cap + constants.O_epsilon:
                print(f"capacity violation: {e} {link_util[e]}/{link_cap}")
                raise Exception("capacity violated")

        for (src, dst) in frozen_flows:
            print(f"demand between {src} and {dst}: {frozen_flows[src, dst]}")

    run_time_dict[MCF_TOTAL] += (datetime.now() - st_time).total_seconds()
    if break_down:
        check_point_retrieve = datetime.now()
        time_model = (check_point_model - st_time_model).total_seconds()
        time_optimize = (check_point_optimize - check_point_model).total_seconds()
        time_retrieve = (check_point_retrieve - check_point_optimize).total_seconds()
        return time_model, time_optimize, time_retrieve
    return None
