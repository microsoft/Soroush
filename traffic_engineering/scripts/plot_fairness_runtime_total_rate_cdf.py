import os, sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from utilities import utils, constants
from alg import waterfilling_utils
from ncflow.lib.problem import Problem


total_flow_file = "../outputs/processed_logs_2022_09_08_01_31_43_0406f92a/total_flow.pickle"

list_methods = [
    constants.APPROX,
    constants.APPROX_BET,
    constants.APPROX_BET_MCF,
    constants.DANNA,
    constants.SWAN,
    constants.ONE_WATERFILLING,
    constants.NEW_APPROX,
]

default_marker_size = 3.5
default_line_width = 3

our_line_width = 3.5
our_marker_size = 3
approach_plot = \
    {
         constants.APPROX: ('#1f77b4', None, ':', our_marker_size, our_line_width),
         # constants.APPROX_MCF:  ('#ff7f0e', None, ':', default_marker_size, default_line_width),
         constants.APPROX_BET: ('#2ca02c', None, ':', our_marker_size, our_line_width),
         constants.APPROX_BET_MCF: ('#d62728', None, "--", our_marker_size, our_line_width),
         constants.APPROX_BET_MCF_BIASED: ('#1a55FF', None, "--", our_marker_size, our_line_width),
         constants.DANNA: ('#9467bd', None, "-.", default_marker_size, default_line_width),
         constants.SWAN: ('#8c564b', None, "-.", default_marker_size, default_line_width),
         constants.ONE_WATERFILLING: ('#e377c2', None, ":", default_marker_size, default_line_width),
         constants.NEW_APPROX: ("#ff7f0e", None, "--", our_marker_size, our_line_width)
    }
default_marker_size = 2.5
default_line_width = 2

our_line_width = 3.5
our_marker_size = 3
approach_plot = \
    {
         constants.APPROX: ('#1f77b4', None, ':', our_marker_size, our_line_width),
         # constants.APPROX_MCF:  ('#ff7f0e', None, ':', default_marker_size, default_line_width),
         constants.APPROX_BET: ('#2ca02c', None, ':', our_marker_size, our_line_width),
         constants.APPROX_BET_MCF: ('#d62728', None, "--", our_marker_size, our_line_width),
         constants.APPROX_BET_MCF_BIASED: ('#1a55FF', None, "--", our_marker_size, our_line_width),
         constants.DANNA: ('#9467bd', None, "-.", default_marker_size, default_line_width),
         constants.SWAN: ('#8c564b', None, "-.", default_marker_size, default_line_width),
         constants.ONE_WATERFILLING: ('#e377c2', None, ":", default_marker_size, default_line_width),
         constants.NEW_APPROX: ("#ff7f0e", None, "--", our_marker_size, our_line_width)
    }

fig_dir = "../figs/"
utils.ensure_dir(fig_dir)

total_flow_baseline = constants.DANNA
fairness_baseline = constants.DANNA
run_time_baseline = constants.SWAN

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

NUM_PATHS_LIST = [16]

TOPO_TM_MODEL_COMB = []
for tm_model in TM_MODEL_LIST:
    for topo_model in TOPO_NAME_LIST:
        TOPO_TM_MODEL_COMB.append((tm_model, topo_model))

plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
plt.rcParams.update(params)

approach_to_total_thru_mapping = utils.read_pickle_file(total_flow_file)
total_flow_baseline_dict = approach_to_total_thru_mapping[total_flow_baseline]

for num_paths in NUM_PATHS_LIST:
    approach_to_normalized_rate_mapping = defaultdict(list)
    for topo_name in TOPO_NAME_LIST:
        for tm_model in TM_MODEL_LIST:
            fnames = utils.find_topo_tm_fname(topo_name, tm_model)
            for fname in fnames:
                file_name = f"'{fname[4]}'"
                baseline_rate = total_flow_baseline_dict[num_paths][file_name]
                for approach in list_methods:
                    approach_to_normalized_rate_mapping[(approach, 0)].append(approach_to_total_thru_mapping[approach][num_paths][file_name] / baseline_rate)

    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    for approach, num_iter in approach_to_normalized_rate_mapping:
        color, marker_style, line_style, marker_size, line_width = approach_plot[approach]
        label = waterfilling_utils.get_approx_label(approach, num_iter)
        sns.ecdfplot(approach_to_normalized_rate_mapping[approach, num_iter], label=label, color=color, marker=marker_style,
                     linestyle=line_style, linewidth=line_width, markersize=marker_size)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.xlabel(f"Total Flow, relative to {total_flow_baseline}")
    plt.ylabel("Fraction of Scenarios")
    plt.title(f"{num_paths} shortest paths")

    min_x = .9
    min_y = .3
    size_arrow = 14
    bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="black", fc="white", lw=2)
    t = plt.text(min_x, min_y, "Better", ha="center", va="center", rotation=0,
                 size=size_arrow,
                 bbox=bbox_props,
                 transform=ax.transAxes)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("rarrow", pad=0.6)
    # plt.annotate("Better", xy=(1.10, 0.3), xytext=(1.03, 0.3),
    #             arrowprops=dict(arrowstyle="->", color="red"), va="center", color="red", fontsize=20, transform=ax.transAxes)
    plt.grid(which="major", axis="x", zorder=-10, linestyle="--")
    plt.grid(which="major", axis="y", zorder=-10, linestyle="--")
    # plt.savefig(fig_dir + f"total_flow_relative_swan_{num_paths}_{utils.get_fid()}.png", bbox_inches='tight', format="png")
    plt.show()
