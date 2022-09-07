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
from scripts import benchmark_plot_utils

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
    constants.APPROX: [("../outputs/approx(1)_2022_08_26_08_25_54_0691655a", 1)],
    constants.APPROX_BET: [("../outputs/approx(1)_bet_2022_08_27_01_50_46_4284934e", 1),
                           ("../outputs/approx(1)_bet_2022_09_06_18_58_51_9f5810ed", 1)],
    constants.APPROX_BET_MCF: [("../outputs/approx(1)_bet(10)_mcf_2022_09_03_12_39_08_e0e2b482", 1),
                               ("../outputs/approx(1)_bet(20)_mcf_2022_09_06_20_44_22_b768b3f8", 1)],
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
        ("computation", False),
        # "extract_rate",
    ],
    constants.APPROX: [
        # "model",
        ("computation", False),
        # "extract_rate",
    ],
    constants.DANNA: [
        # "model",
        ("feasibility_total", False),
        # "feasibility_solver",
        ("mcf_total", False),
        # "mcf_solver",
        # 'extract_rate',
    ],
    constants.SWAN: [
        # 'model',
        ('mcf_total', False),
        # 'mcf_solver',
        # 'freeze_time'
    ],
    constants.NEW_APPROX: [
        # 'model',
        # 'extract_rate',
        ('mcf_solver', False),
    ],
    constants.APPROX_BET: [
        # "model",
        ("computation", False),
        # "extract_rate",
    ],
    constants.APPROX_BET_MCF: [
        # "model_approx_bet_p_mcf",
        # "extract_rate_approx_bet_p_mcf",
        # "model",
        ("computation", False),
        # "extract_rate",
        # "model_equi_depth",
        ("solver_time_equi_depth", True),
        # "extract_rate_equi_depth",
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
    # 'Uninett2010.graphml',
    'Cogentco.graphml',
    'GtsCe.graphml',
    'UsCarrier.graphml',
    # 'Colt.graphml',
    'TataNld.graphml',
    # 'Kdl.graphml',
]

APPROX_TOPO_NAME_TO_ITERATION = {
    "'Uninett2010.graphml'": (1, 10),
    "'Cogentco.graphml'": (1, 10),
    "'GtsCe.graphml'": (1, 10),
    "'UsCarrier.graphml'": (1, 20),
    "'Colt.graphml'": (1, 10),
    "'TataNld.graphml'": (1, 20),
    # 'Kdl.graphml',
}

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
    # print(approach, approach_plot)
    assert approach in approach_plot
# plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
plt.rcParams.update(params)

approach_to_total_thru_mapping = defaultdict(lambda: defaultdict(dict))
approach_to_run_time_mapping = defaultdict(lambda: defaultdict(dict))
approach_to_fid_to_rate_fname_mapping = defaultdict(lambda: defaultdict(dict))
for approach, dir_list in approach_to_log_dir_mapping.items():
    output = benchmark_plot_utils.read_rate_log_file(approach, dir_list,
                                                     valid_topo_traffic_list=TOPO_TM_MODEL_COMB,
                                                     approach_to_valid_for_run_time=approach_to_valid_for_run_time,
                                                     topo_name_to_approx_param=APPROX_TOPO_NAME_TO_ITERATION)
    approach_to_total_thru_mapping[approach] = output[0]
    approach_to_run_time_mapping[approach] = output[1]
    approach_to_fid_to_rate_fname_mapping[approach] = output[2]


# cdf_1 = []
# cdf_2 = []
# for name_file in approach_to_run_time_mapping[constants.APPROX_BET_MCF][16]:
#     scale_factor = float(name_file.split("/")[-1].split("_")[3])
#     if scale_factor <= 6.9:
#         print(f"===== {name_file}")
#         baseline_run_time = approach_to_run_time_mapping[constants.SWAN][16][name_file]
#         print(approach_to_run_time_mapping[constants.APPROX_BET_MCF][16][name_file]['1', '10'] / baseline_run_time)
#         cdf_1.append(baseline_run_time / approach_to_run_time_mapping[constants.APPROX_BET_MCF][16][name_file]['1', '10'])
#         print(approach_to_run_time_mapping[constants.APPROX_BET][16][name_file]['1', '10'] / baseline_run_time)
#         cdf_2.append(baseline_run_time / approach_to_run_time_mapping[constants.APPROX_BET][16][name_file]['1', '10'])
# print(len(cdf_1))
# sns.ecdfplot(cdf_1, label="1")
# sns.ecdfplot(cdf_2, label="2")
# plt.legend()
# plt.xscale('log')
# plt.show()
# raise Exception


# approach_baseline = constants.SWAN
# approach_compare = constants.APPROX_BET_MCF
# approach_iter = ('1', '10')
# assert approach_baseline in approach_to_log_dir_mapping
# assert approach_compare in approach_to_log_dir_mapping
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
            # if is_approx:
                # for num_iter, approx_rate in approach_to_total_thru_mapping[approach][num_paths][file_name].items():
            approach_to_normalized_rate_mapping[(approach, 0)].append(approach_to_total_thru_mapping[approach][num_paths][file_name] / baseline_rate)
            # else:
            #     approach_to_normalized_rate_mapping[(approach, 0)].append(approach_to_total_thru_mapping[approach][num_paths][file_name] / baseline_rate)

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
        scale_factor = float(file_name.split("/")[-1].split("_")[3])
        if scale_factor > 6.9:
            continue
        baseline_run_time = run_time_baseline_dict[num_paths][file_name]
        for approach in approach_to_log_dir_mapping:
            is_approx = approach_to_log_dir_mapping[approach][0][1]
            # if is_approx:
            #     for num_iter, run_time in approach_to_run_time_mapping[approach][num_paths][file_name].items():
            #         approach_to_normalized_run_time[(approach, num_iter)].append(baseline_run_time / run_time)
            # else:
            approach_to_normalized_run_time[(approach, 0)].append(baseline_run_time / approach_to_run_time_mapping[approach][num_paths][file_name])

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
    plt.xscale('log')
    plt.title(f"{num_paths} shortest paths")
    plt.savefig(fig_dir + f"speedup_relative_swan_{num_paths}_{utils.get_fid()}.png", bbox_inches='tight', format="png")


fairness_baseline_fname_dict = approach_to_fid_to_rate_fname_mapping[fairness_baseline]

print("======================== Fairness Analysis")
for num_paths in NUM_PATH_LIST:
    approach_to_fairness = defaultdict(list)
    for file_name in fairness_baseline_fname_dict[num_paths]:
        baseline_fid_to_rate_mapping = utils.read_pickle_file(fairness_baseline_fname_dict[num_paths][file_name])
        for approach in approach_to_log_dir_mapping:
            print(f"=== File Name {file_name} num paths {num_paths} approach {approach}")
            is_approx = approach_to_log_dir_mapping[approach][0][1]
            # if is_approx:
            #     for num_iter, fid_to_rate_mapping_fname in approach_to_fid_to_rate_fname_mapping[approach][num_paths][file_name].items():
            # fid_to_rate_mapping = utils.read_pickle_file(fid_to_rate_mapping_fname)
            fid_to_rate_mapping = utils.read_pickle_file(approach_to_fid_to_rate_fname_mapping[approach][num_paths][file_name])
            fairness_no = benchmark_plot_utils.compute_fairness_no(baseline_fid_to_rate_mapping,
                                                                   fid_to_rate_mapping, theta_fairness)
            approach_to_fairness[(approach, 0)].append(fairness_no)
            print(approach, fairness_no)
            # else:
            #     fid_to_rate_mapping = utils.read_pickle_file(approach_to_fid_to_rate_fname_mapping[approach][num_paths][file_name])
            #     approach_to_fairness[(approach, 0)].append(benchmark_plot_utils.compute_fairness_no(baseline_fid_to_rate_mapping,
            #                                                                                         fid_to_rate_mapping,
            #                                                                                         theta_fairness))

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

