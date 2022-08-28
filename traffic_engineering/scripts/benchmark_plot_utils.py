from collections import defaultdict

import numpy as np

from utilities import utils


def is_topo_traffic_valid(fname, valid_topo_traffic_list):
    if valid_topo_traffic_list == "*":
        return True
    for topo_name, tm_name in valid_topo_traffic_list:
        if topo_name in fname and tm_name in fname:
            return True
    return False


def get_output_run_time(approach_name, run_time_dict, approach_to_valid_for_run_time):
    run_time = 0
    if approach_to_valid_for_run_time[approach_name] == "*":
        run_time = sum([r_time for _, r_time in run_time_dict.items()])
        return run_time

    for time_name in approach_to_valid_for_run_time[approach_name]:
        run_time += run_time_dict[time_name]
    return run_time


def parse_line(line, dir_name, approach_name, is_approx, approach_to_valid_for_run_time):
    model_param, other_param = line.split(")")
    flow_rate_file_name = dir_name + "/{}_per_path_flow.pickle"
    run_time_dict_file_name = dir_name + "/{}_run_time_dict.pickle"

    model_param = model_param[1:]
    print(line, dir_name)
    topo_name, traffic_name, scale_factor, topo_file, traffic_file = model_param.split(", ")

    other_param = other_param[1:]
    if is_approx:
        try:
            num_paths, num_iter_approx, total_flow, run_time, detailed_per_flow_file = other_param.split(",")
            num_iter = (num_iter_approx)
        except ValueError:
            num_paths, num_iter_approx, num_iter_bet, total_flow, run_time, detailed_per_flow_file = other_param.split(",")
            num_iter = (num_iter_approx, num_iter_bet)
        print(detailed_per_flow_file)
        fid_to_rate_mapping = utils.read_pickle_file(flow_rate_file_name.format(detailed_per_flow_file[1:-1]))
        run_time_dict = utils.read_pickle_file(run_time_dict_file_name.format(detailed_per_flow_file[1:-1]))
        output_run_time = get_output_run_time(approach_name, run_time_dict, approach_to_valid_for_run_time)
        return traffic_file, int(num_paths), num_iter, float(total_flow), output_run_time, fid_to_rate_mapping

    num_paths, total_flow, run_time, detailed_per_flow_file = other_param.split(",")
    fid_to_rate_mapping = utils.read_pickle_file(flow_rate_file_name.format(detailed_per_flow_file[:-1]))
    run_time_dict = utils.read_pickle_file(run_time_dict_file_name.format(detailed_per_flow_file[:-1]))
    output_run_time = get_output_run_time(approach_name, run_time_dict, approach_to_valid_for_run_time)
    return traffic_file, int(num_paths), float(total_flow), output_run_time, fid_to_rate_mapping


def get_total_thru_run_time_dict_approx(approach_name, dir_name, valid_topo_traffic_list, approach_to_valid_for_run_time):
    total_thru_dict = defaultdict(lambda: defaultdict(dict))
    run_time_dict = defaultdict(lambda: defaultdict(dict))
    fid_to_flow_rate_mapping_dict = defaultdict(lambda: defaultdict(dict))
    with open(dir_name + ".txt", "r") as fp:
        for l in fp.readlines():
            output_line = parse_line(l, dir_name, approach_name, True, approach_to_valid_for_run_time)
            traffic_file, num_paths, num_iter, total_flow, run_time, fid_to_rate_mapping = output_line
            if not is_topo_traffic_valid(traffic_file, valid_topo_traffic_list):
                print(f"skipping ${traffic_file}$ either topo or traffic not valid!!")
                continue
            total_thru_dict[num_paths][traffic_file][num_iter] = total_flow
            run_time_dict[num_paths][traffic_file][num_iter] = run_time
            fid_to_flow_rate_mapping_dict[num_paths][traffic_file][num_iter] = fid_to_rate_mapping
    return total_thru_dict, run_time_dict, fid_to_flow_rate_mapping_dict


def get_total_thru_run_time_dict(approach_name, dir_name, valid_topo_traffic_list, approach_to_valid_for_run_time):
    total_thru_dict = defaultdict(dict)
    run_time_dict = defaultdict(dict)
    fid_to_flow_rate_mapping_dict = defaultdict(dict)
    with open(dir_name + ".txt", "r") as fp:
        for l in fp.readlines():
            output_line = parse_line(l, dir_name, approach_name, False, approach_to_valid_for_run_time)
            traffic_file, num_paths, total_flow, run_time, fid_to_rate_mapping = output_line
            if not is_topo_traffic_valid(traffic_file, valid_topo_traffic_list):
                print(f"skipping either topo or traffic not valid!!")
                continue
            total_thru_dict[num_paths][traffic_file] = total_flow
            run_time_dict[num_paths][traffic_file] = run_time
            fid_to_flow_rate_mapping_dict[num_paths][traffic_file] = fid_to_rate_mapping
    return total_thru_dict, run_time_dict, fid_to_flow_rate_mapping_dict


def read_rate_log_file(approach, dir_list, valid_topo_traffic_list, approach_to_valid_for_run_time):
    approach_to_total_thru_mapping = defaultdict(dict)
    approach_to_run_time_mapping = defaultdict(dict)
    approach_to_fid_to_rate_mapping = defaultdict(dict)
    for dir_name, is_approx in dir_list:
        if is_approx:
            total_thru_run_time_dict = get_total_thru_run_time_dict_approx(approach, dir_name,
                                                                           valid_topo_traffic_list,
                                                                           approach_to_valid_for_run_time)
            for num_paths in total_thru_run_time_dict[0]:
                for traffic_file in total_thru_run_time_dict[0][num_paths]:
                    for num_iter in total_thru_run_time_dict[0][num_paths][traffic_file]:
                        if traffic_file not in approach_to_total_thru_mapping[num_paths] or \
                                num_iter not in approach_to_total_thru_mapping[num_paths][traffic_file]:
                            approach_to_total_thru_mapping[num_paths][traffic_file] = dict()
                            approach_to_run_time_mapping[num_paths][traffic_file] = dict()
                            approach_to_fid_to_rate_mapping[num_paths][traffic_file] = dict()
                        approach_to_total_thru_mapping[num_paths][traffic_file][num_iter] = \
                            total_thru_run_time_dict[0][num_paths][traffic_file][num_iter]
                        approach_to_run_time_mapping[num_paths][traffic_file][num_iter] = \
                            total_thru_run_time_dict[1][num_paths][traffic_file][num_iter]
                        approach_to_fid_to_rate_mapping[num_paths][traffic_file][num_iter] = \
                            total_thru_run_time_dict[2][num_paths][traffic_file][num_iter]
        else:
            total_thru_run_time_dict = get_total_thru_run_time_dict(approach, dir_name,
                                                                    valid_topo_traffic_list,
                                                                    approach_to_valid_for_run_time)
            for num_paths in total_thru_run_time_dict[0]:
                for traffic_file in total_thru_run_time_dict[0][num_paths]:
                    approach_to_total_thru_mapping[num_paths][traffic_file] = \
                        total_thru_run_time_dict[0][num_paths][traffic_file]
                    approach_to_run_time_mapping[num_paths][traffic_file] = \
                        total_thru_run_time_dict[1][num_paths][traffic_file]
                    approach_to_fid_to_rate_mapping[num_paths][traffic_file] = \
                        total_thru_run_time_dict[2][num_paths][traffic_file]
    return approach_to_total_thru_mapping, approach_to_run_time_mapping, approach_to_fid_to_rate_mapping


def compute_fairness_no(baseline_mapping, approach_mapping, theta_fairness):
    fairness_num = 0
    num_flows = 0
    for fid in baseline_mapping:
        if np.sum(baseline_mapping[fid]) <= 0:
            continue
        assert fid in approach_mapping
        baseline_rate = np.sum(baseline_mapping[fid])
        alg_rate = np.sum(approach_mapping[fid])
        f_1 = max(theta_fairness, alg_rate) / max(theta_fairness, baseline_rate)
        f_2 = max(baseline_rate, theta_fairness) / max(theta_fairness, alg_rate)
        fairness_num += np.log10(min(f_1, f_2))
        num_flows += 1
    return np.power(10, fairness_num/num_flows)


def compute_fairness_no_vectorized_baseline(baseline_rate_vector, approach_mapping, list_commodities, theta_fairness):
    assert len(baseline_rate_vector) == len(approach_mapping)
    assert np.all(baseline_rate_vector > 0)
    approach_vectorized = np.zeros(shape=len(baseline_rate_vector))
    for fid, (src, dst, _) in list_commodities:
        approach_vectorized[fid] = np.sum(approach_mapping[(src, dst)])

    quantized_approach_vector = np.maximum(approach_vectorized, theta_fairness)
    quantized_baseline_vector = np.maximum(baseline_rate_vector, theta_fairness)

    output_fairness_1 = quantized_baseline_vector / quantized_approach_vector
    output_fairness_2 = quantized_approach_vector / quantized_baseline_vector

    assert not np.any(np.isnan(output_fairness_1))
    assert not np.any(np.isnan(output_fairness_2))
    fairness_no = np.sum(np.log10(np.minimum(output_fairness_1, output_fairness_2)))

    return np.power(10, fairness_no / len(baseline_rate_vector))
