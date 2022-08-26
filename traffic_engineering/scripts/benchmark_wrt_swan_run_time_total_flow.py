import os, sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cycler import cycler
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from utilities import utils, constants
from alg import waterfilling_utils

# approx_log_file = "../outputs/approx_2022_03_28_09_18_01_96a80821.txt"
# # one_water_log_file = "../outputs/one_water_2022_03_28_10_13_45_a822908a.txt"
# one_water_log_file = "../outputs/one_water_2022_03_28_16_25_06_8f090185.txt"
# swan_log_file = "../outputs/swan_2022_03_28_10_42_02_44c512d6.txt"

approach_to_log_dir_mapping = {
    # constants.APPROX: [("../outputs/approx_2022_04_08_01_56_20_9c3cc88d", 1)],
    # constants.APPROX_MCF: [("../outputs/approx_mcf_2022_04_08_18_28_40_528acb29", 1)],
    # constants.APPROX_BET: [("../outputs/approx_bet_2022_04_10_10_12_57_f694013c", 1)],
    # constants.APPROX_BET_MCF_BIASED: [("../outputs/approx_bet_biased_mcf_2022_04_28_22_05_41_992b96ba", 1)],
    # constants.APPROX_BET_MCF: [("../outputs/approx_bet_mcf_2022_04_10_12_49_24_b64bd9fb", 1)],
    # constants.APPROX_BET_MCF: [("../outputs/approx_bet_mcf_2022_05_15_16_13_34_987c39c9", 1)],
    # constants.NEW_APPROX: [("../outputs/new_approx_2022_07_10_12_18_18_09a71814", 0)],
    # constants.APPROX_BET_MCF: [("../outputs/approx(1)_bet(10)_mcf_2022_05_16_04_20_06_14cf9ecc", 1),
    #                            ("../outputs/approx(1)_bet(10)_mcf_2022_05_22_23_36_38_ecf2c718", 1)],
    # constants.DANNA: [("../outputs/danna_practical_2022_04_05_11_21_52_5a365c0f", 0),
    #                   ("../outputs/danna_practical_2022_04_08_02_34_55_ed6031ff", 0),
    #                   ("../outputs/danna_practical_2022_05_17_10_02_17_b535e7d3", 0),
    #                   ("../outputs/danna_practical_2022_05_20_20_22_39_63efe481", 0),
    #                   ("../outputs/danna_practical_2022_05_21_18_46_44_911d1fa3", 0),
    #                   ("../outputs/danna_practical_2022_05_22_09_17_34_b0c9cdeb", 0)],
    # constants.SWAN: [("../outputs/swan_2022_04_07_10_22_17_d9a42dd7", 0),
    #                  ("../outputs/swan_2022_04_08_13_50_46_6bdfa84d", 0),
    #                  ("../outputs/swan_2022_04_09_23_17_04_1af420da", 0),
    #                  ("../outputs/swan_2022_05_21_08_57_09_81eae09f", 0)],
    constants.NEW_APPROX: [("../outputs/geometric_binner_2022_08_25_06_02_02_72f6231f", 0)],
    constants.DANNA: [("../outputs/danna_practical_2022_08_16_17_13_19_dc44cf7e", 0),
                      ("../outputs/danna_practical_2022_08_16_22_22_21_6afe1d48", 0)],
    constants.SWAN: [("../outputs/swan_2022_08_23_07_06_01_47c2318f", 0)],
    constants.ONE_WATERFILLING: [("../outputs/one_water_filling_2022_08_26_03_45_36_08443d49", 0)]
    # constants.ONE_WATERFILLING: [("../outputs/one_water_2022_04_08_00_28_28_45cc389d", 0),
    #                              ("../outputs/one_water_2022_04_08_15_37_42_4d1f63ed", 0)]
}

approach_to_valid_for_run_time = {
    constants.ONE_WATERFILLING: [
        # "model",
        "computation",
        # "extract_rate",
    ],
    constants.APPROX: [
        # "model",
        "computation",
        # "extract_rate",
    ],
    constants.DANNA: [
        # "model",
        "feasibility_total",
        # "feasibility_solver",
        "mcf_total",
        # "mcf_solver",
        # 'extract_rate',
    ],
    constants.SWAN: [
        # 'model',
        'mcf_total',
        # 'mcf_solver',
        # 'freeze_time'
    ],
    constants.NEW_APPROX: [
        # 'model',
        # 'extract_rate',
        'mcf_solver',
    ]
}

fig_dir = "../figs/"
utils.ensure_dir(fig_dir)

total_flow_baseline = constants.DANNA
assert total_flow_baseline in approach_to_log_dir_mapping
fairness_baseline = constants.DANNA
assert fairness_baseline in approach_to_log_dir_mapping
run_time_baseline = constants.SWAN
assert run_time_baseline in approach_to_log_dir_mapping

theta_fairness = 0.1

TM_MODEL_LIST = [
    'uniform',
    'bimodal',
    'gravity',
    'poisson-high-inter',
    # 'poisson-high-intra',
]
TOPO_NAME_LIST = [
    'Uninett2010.graphml',
    'Cogentco.graphml',
    'GtsCe.graphml',
    'UsCarrier.graphml',
    'Colt.graphml',
    'TataNld.graphml',
    # 'Kdl.graphml',
]

TOPO_TM_MODEL_COMB = []
for tm_model in TM_MODEL_LIST:
    for topo_model in TOPO_NAME_LIST:
        TOPO_TM_MODEL_COMB.append((tm_model, topo_model))


NUM_PATH_LIST = [16]
TOPO_EDGE_NUM = [202, 386, 486]
topo_to_num_edges = {'Uninett2010.graphml': 202,
                     'GtsCe.graphml': 386,
                     'Cogentco.graphml': 486}

default_marker_size = 1.5
default_line_width = 2
approach_plot = {constants.APPROX: ('#1f77b4', None, ':', default_marker_size, default_line_width),
                 # constants.APPROX_MCF:  ('#ff7f0e', None, ':', default_marker_size, default_line_width),
                 constants.APPROX_BET: ('#2ca02c', None, ':', default_marker_size, default_line_width),
                 constants.APPROX_BET_MCF: ('#d62728', None, "--", 3, 2),
                 constants.APPROX_BET_MCF_BIASED: ('#1a55FF', None, "--", 3, 2),
                 constants.DANNA: ('#9467bd', None, "-.", 2.5, 2),
                 constants.SWAN: ('#8c564b', None, "-.", 2.5, 2),
                 constants.ONE_WATERFILLING: ('#e377c2', None, ":", default_marker_size, default_line_width),
                 constants.NEW_APPROX: ("#ff7f0e", None, "--", 3, 2)}
                  # '#7f7f7f',
                  # '#bcbd22',
                  # '#17becf',
                  # '#1a55FF'}

for approach in approach_to_log_dir_mapping:
    assert approach in approach_plot
# plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
plt.rcParams.update(params)

def get_output_run_time(approach_name, run_time_dict):
    run_time = 0
    for time_name in approach_to_valid_for_run_time[approach_name]:
        run_time += run_time_dict[time_name]
    return run_time


def parse_line(line, dir_name, approach_name, is_approx):
    model_param, other_param = line.split(")")
    flow_rate_file_name = dir_name + "/{}_per_flow_rate.pickle"
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
        output_run_time = get_output_run_time(approach_name, run_time_dict)
        return traffic_file, int(num_paths), num_iter, float(total_flow), output_run_time, fid_to_rate_mapping

    num_paths, total_flow, run_time, detailed_per_flow_file = other_param.split(",")
    fid_to_rate_mapping = utils.read_pickle_file(flow_rate_file_name.format(detailed_per_flow_file[:-1]))
    run_time_dict = utils.read_pickle_file(run_time_dict_file_name.format(detailed_per_flow_file[:-1]))
    output_run_time = get_output_run_time(approach_name, run_time_dict)
    return traffic_file, int(num_paths), float(total_flow), output_run_time, fid_to_rate_mapping


def get_total_thru_run_time_dict_approx(approach_name, dir_name):
    total_thru_dict = defaultdict(lambda: defaultdict(dict))
    run_time_dict = defaultdict(lambda: defaultdict(dict))
    fid_to_flow_rate_mapping_dict = defaultdict(lambda: defaultdict(dict))
    with open(dir_name + ".txt", "r") as fp:
        for l in fp.readlines():
            traffic_file, num_paths, num_iter, total_flow, run_time, fid_to_rate_mapping = parse_line(l, dir_name,
                                                                                                      approach_name,
                                                                                                      is_approx=True)
            if not is_topo_traffic_valid(traffic_file):
                print(f"skipping ${traffic_file}$ either topo or traffic not valid!!")
                continue
            total_thru_dict[num_paths][traffic_file][num_iter] = total_flow
            run_time_dict[num_paths][traffic_file][num_iter] = run_time
            fid_to_flow_rate_mapping_dict[num_paths][traffic_file][num_iter] = fid_to_rate_mapping
    return total_thru_dict, run_time_dict, fid_to_flow_rate_mapping_dict


def get_total_thru_run_time_dict(approach_name, dir_name):
    total_thru_dict = defaultdict(dict)
    run_time_dict = defaultdict(dict)
    fid_to_flow_rate_mapping_dict = defaultdict(dict)
    with open(dir_name + ".txt", "r") as fp:
        for l in fp.readlines():
            traffic_file, num_paths, total_flow, run_time, fid_to_rate_mapping = parse_line(l, dir_name,
                                                                                            approach_name,
                                                                                            is_approx=False)
            if not is_topo_traffic_valid(traffic_file):
                print(f"skipping either topo or traffic not valid!!")
                continue
            total_thru_dict[num_paths][traffic_file] = total_flow
            run_time_dict[num_paths][traffic_file] = run_time
            fid_to_flow_rate_mapping_dict[num_paths][traffic_file] = fid_to_rate_mapping
    return total_thru_dict, run_time_dict, fid_to_flow_rate_mapping_dict


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


def is_topo_traffic_valid(fname):
    for topo_name, tm_name in TOPO_TM_MODEL_COMB:
        if topo_name in fname and tm_name in fname:
            return True
    return False
    # topo_valid = False
    # for topo in TOPO_NAME_LIST:
    #     if topo in fname:
    #         topo_valid = True
    #         break
    #
    # traffic_valid = False
    # for traffic_model in TM_MODEL_LIST:
    #     if traffic_model in fname:
    #         traffic_valid = True
    #         break
    #
    # return traffic_valid and topo_valid


approach_to_total_thru_mapping = defaultdict(lambda: defaultdict(dict))
approach_to_run_time_mapping = defaultdict(lambda: defaultdict(dict))
approach_to_fid_to_rate_mapping = defaultdict(lambda: defaultdict(dict))
for approach, dir_list in approach_to_log_dir_mapping.items():
    for dir, is_approx in dir_list:
        if is_approx:
            output = get_total_thru_run_time_dict_approx(approach, dir)
            for num_paths in output[0]:
                for traffic_file in output[0][num_paths]:
                    for num_iter in output[0][num_paths][traffic_file]:
                        if traffic_file not in approach_to_total_thru_mapping[approach][num_paths] or \
                                num_iter not in approach_to_total_thru_mapping[approach][num_paths][traffic_file]:
                            approach_to_total_thru_mapping[approach][num_paths][traffic_file] = dict()
                            approach_to_run_time_mapping[approach][num_paths][traffic_file] = dict()
                            approach_to_fid_to_rate_mapping[approach][num_paths][traffic_file] = dict()
                        approach_to_total_thru_mapping[approach][num_paths][traffic_file][num_iter] = output[0][num_paths][traffic_file][num_iter]
                        approach_to_run_time_mapping[approach][num_paths][traffic_file][num_iter]  = output[1][num_paths][traffic_file][num_iter]
                        approach_to_fid_to_rate_mapping[approach][num_paths][traffic_file][num_iter]  = output[2][num_paths][traffic_file][num_iter]
        else:
            output = get_total_thru_run_time_dict(approach, dir)
            for num_paths in output[0]:
                for traffic_file in output[0][num_paths]:
                    approach_to_total_thru_mapping[approach][num_paths][traffic_file] = output[0][num_paths][traffic_file]
                    approach_to_run_time_mapping[approach][num_paths][traffic_file] = output[1][num_paths][traffic_file]
                    approach_to_fid_to_rate_mapping[approach][num_paths][traffic_file] = output[2][num_paths][traffic_file]



approach_baseline = constants.SWAN
approach_compare = constants.APPROX_BET_MCF
approach_iter = ('1', '10')
assert approach_baseline in approach_to_log_dir_mapping
assert approach_compare in approach_to_log_dir_mapping
# '../ncflow/traffic-matrices/bimodal/Uninett2010.graphml_bimodal_987377064_16.0_0.2_0.1-0.2_0.4-0.8_traffic-matrix.pkl'
# run_time_baseline_dict = approach_to_run_time_mapping[approach_baseline]
# for num_paths in run_time_baseline_dict:
#     speedup_heatmap = np.zeros(shape=(8, 3))
#     num_each = defaultdict(int)
#     speedup_dict = defaultdict(lambda: pd.DataFrame(speedup_heatmap, index=[1, 2, 4, 8, 16, 32, 64, 128], columns=TOPO_NAME_LIST).copy())
#     for file_name in run_time_baseline_dict[num_paths]:
#         topo_name = file_name.split("/")[4].split("_")[0]
#         scale_factor = int(float(file_name.split("/")[4].split("_")[3]))
#         traffic_model = file_name.split("/")[4].split("_")[1]
#         baseline_run_time = run_time_baseline_dict[num_paths][file_name]
#         for approach in approach_to_log_dir_mapping:
#             if approach != approach_compare:
#                 continue
#             is_approx = approach_to_log_dir_mapping[approach][0][1]
#             if is_approx:
#                 for num_iter, run_time in approach_to_run_time_mapping[approach][num_paths][file_name].items():
#                     if num_iter != approach_iter:
#                         continue
#                     speedup_dict[traffic_model].at[scale_factor, topo_name] += baseline_run_time / run_time
#             else:
#                 speedup_dict[traffic_model].at[scale_factor, topo_name] += (baseline_run_time / approach_to_run_time_mapping[approach][num_paths][file_name])
#             num_each[traffic_model, scale_factor, topo_name] += 1
#
#     print(speedup_dict)
#     print(num_each)
#     for key in num_each:
#         print(key)
#         speedup_dict[key[0]].at[key[1], key[2]] = speedup_dict[key[0]].at[key[1], key[2]] / num_each[key]
#
#     for traffic_model in speedup_dict:
#         plt.figure(figsize=(7, 5))
#         speedup_dict[traffic_model].columns = TOPO_EDGE_NUM
#         sns.heatmap(speedup_dict[traffic_model], annot=True, fmt=".1f")
#         # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
#         plt.ylabel("Scale factor")
#         plt.xlabel("#Edges")
#         plt.title(f"Average speed up ({traffic_model}, {num_paths} paths)")
#         plt.savefig(fig_dir + f"speedup_relative_swan_{traffic_model}_{num_paths}_heatmap.png", bbox_inches='tight', format="png")
#         # plt.show()
#
# run_time_baseline_dict = approach_to_run_time_mapping[run_time_baseline]
# for num_paths in run_time_baseline_dict:
#     for file_name in run_time_baseline_dict[num_paths]:
#         baseline_run_time = run_time_baseline_dict[num_paths][file_name]
#         for approach in approach_to_log_dir_mapping:
#             is_approx = approach_to_log_dir_mapping[approach][0][1]
#             if is_approx:
#                 for num_iter, run_time in approach_to_run_time_mapping[approach][num_paths][file_name].items():
#                     if baseline_run_time < run_time:
#                         print(file_name, run_time_baseline, baseline_run_time, approach, run_time, num_paths)

total_flow_baseline_dict = approach_to_total_thru_mapping[total_flow_baseline]
for num_paths in NUM_PATH_LIST:
    approach_to_normalized_rate_mapping = defaultdict(list)
    for file_name in total_flow_baseline_dict[num_paths]:
        baseline_rate = total_flow_baseline_dict[num_paths][file_name]
        for approach in approach_to_log_dir_mapping:
            is_approx = approach_to_log_dir_mapping[approach][0][1]
            if is_approx:
                for num_iter, approx_rate in approach_to_total_thru_mapping[approach][num_paths][file_name].items():
                    approach_to_normalized_rate_mapping[(approach, num_iter)].append(approx_rate / baseline_rate)
            else:
                approach_to_normalized_rate_mapping[(approach, 0)].append(approach_to_total_thru_mapping[approach][num_paths][file_name] / baseline_rate)

    plt.figure(figsize=(9, 5))
    for approach, num_iter in approach_to_normalized_rate_mapping:
        color, marker_style, line_style, marker_size, line_width = approach_plot[approach]
        label = waterfilling_utils.get_approx_label(approach, num_iter)
        sns.ecdfplot(approach_to_normalized_rate_mapping[approach, num_iter], label=label, color=color, marker=marker_style,
                     linestyle=line_style, linewidth=line_width, markersize=marker_size)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.xlabel(f"Total Flow, relative to {total_flow_baseline}")
    plt.ylabel("Fraction of Scenarios")
    plt.title(f"{num_paths} shortest paths")
    plt.savefig(fig_dir + f"total_flow_relative_swan_{num_paths}_{utils.get_fid()}.png", bbox_inches='tight', format="png")


run_time_baseline_dict = approach_to_run_time_mapping[run_time_baseline]
for num_paths in NUM_PATH_LIST:
    approach_to_normalized_run_time = defaultdict(list)
    for file_name in run_time_baseline_dict[num_paths]:
        baseline_run_time = run_time_baseline_dict[num_paths][file_name]
        for approach in approach_to_log_dir_mapping:
            is_approx = approach_to_log_dir_mapping[approach][0][1]
            if is_approx:
                for num_iter, run_time in approach_to_run_time_mapping[approach][num_paths][file_name].items():
                    approach_to_normalized_run_time[(approach, num_iter)].append(np.log10(baseline_run_time / run_time))
            else:
                approach_to_normalized_run_time[(approach, 0)].append(np.log10(baseline_run_time / approach_to_run_time_mapping[approach][num_paths][file_name]))

    plt.figure(figsize=(9, 5))
    for approach, num_iter in approach_to_normalized_run_time:
        # ('#1f77b4', None, ':', default_marker_size)
        color, marker_style, line_style, marker_size, line_width = approach_plot[approach]
        label = waterfilling_utils.get_approx_label(approach, num_iter)
        sns.ecdfplot(approach_to_normalized_run_time[approach, num_iter], label=label, color=color, marker=marker_style,
                     linestyle=line_style, linewidth=line_width, markersize=marker_size)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.xlabel(f"Speedup, relative to {run_time_baseline} (log scale)")
    plt.ylabel("Fraction of Scenarios")
    plt.title(f"{num_paths} shortest paths")
    plt.savefig(fig_dir + f"speedup_relative_swan_{num_paths}_{utils.get_fid()}.png", bbox_inches='tight', format="png")


fairness_baseline_dict = approach_to_fid_to_rate_mapping[fairness_baseline]

print("======================== Fairness Analysis")
for num_paths in NUM_PATH_LIST:
    approach_to_fairness = defaultdict(list)
    for file_name in fairness_baseline_dict[num_paths]:
        baseline_fid_to_rate_mapping = fairness_baseline_dict[num_paths][file_name]
        for approach in approach_to_log_dir_mapping:
            print(f"=== File Name {file_name} num paths {num_paths} approach {approach}")
            is_approx = approach_to_log_dir_mapping[approach][0][1]
            if is_approx:
                for num_iter, fid_to_rate_mapping in approach_to_fid_to_rate_mapping[approach][num_paths][file_name].items():
                    fairness_no = compute_fairness_no(baseline_fid_to_rate_mapping, fid_to_rate_mapping, theta_fairness)
                    approach_to_fairness[(approach, num_iter)].append(fairness_no)
            else:
                approach_to_fairness[(approach, 0)].append(compute_fairness_no(baseline_fid_to_rate_mapping,
                                                                          approach_to_fid_to_rate_mapping[approach][num_paths][file_name],
                                                                          theta_fairness))

    plt.figure(figsize=(9, 5))
    for approach, num_iter in approach_to_fairness:
        color, marker_style, line_style, marker_size, line_width = approach_plot[approach]
        label = waterfilling_utils.get_approx_label(approach, num_iter)
        sns.ecdfplot(approach_to_fairness[approach, num_iter], label=label, color=color, marker=marker_style,
                     linestyle=line_style, linewidth=line_width, markersize=marker_size)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.xlabel("Fairness Number")
    plt.ylabel("Fraction of Scenarios")
    plt.title(f"{num_paths} shortest paths")
    plt.savefig(fig_dir + f"fairness_relative_danna_{num_paths}_{utils.get_fid()}.png", bbox_inches='tight', format="png")

