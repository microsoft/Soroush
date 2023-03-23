import os
from collections import defaultdict
from datetime import datetime
from glob import iglob

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from alg import approx_waterfilling
from utilities import shortest_paths, constants
from alg import swan_max_min_approx
from utilities import utils
from ncflow.lib.problem import Problem
from alg import k_waterfilling
#, k_water_bet
from alg import danna_practical_max_min_fair
from alg import iewf_upward_max_min
from alg import approx_water_plus_mcf, approx_water_bet
from alg import approx_water_bet_plus_mcf
from alg import geometric_approx_binning
from scripts import benchmark_plot_utils

#TM_MODEL = 'poisson-high-inter'
# TM_MODEL = 'gravity'
TM_MODEL = 'uniform'

# TOPO_NAME = 'Kdl.graphml'
# TOPO_NAME = 'Cogentco.graphml'
# TOPO_NAME = 'GtsCe.graphml'
# TOPO_NAME = 'UsCarrier.graphml'
# TOPO_NAME = 'Colt.graphml'
# TOPO_NAME = 'Ion.graphml'
TOPO_NAME = 'Uninett2010.graphml'
LOG_DIR = f"./outputs/examples/{TOPO_NAME}/{TM_MODEL}"
utils.ensure_dir(LOG_DIR)
f_idx = 38


def find_topo_tm_fname(topo_name, tm_model):
    fname_list = []
    topo_fname = os.path.join('.', 'ncflow', 'topologies', 'topology-zoo', topo_name)
    for tm_fname in iglob(
            './ncflow/traffic-matrices/{}/{}*_traffic-matrix.pkl'.format(
                tm_model, topo_name)):
        vals = os.path.basename(tm_fname)[:-4].split('_')
        _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
        fname_list.append((topo_name, tm_model, scale_factor, topo_fname, tm_fname))
    return fname_list


fnames = find_topo_tm_fname(TOPO_NAME, TM_MODEL)
num_paths_per_flow = 16

num_iter_bet = 10
# file_name = fnames[f_idx]
for file_name in fnames:
    if "Uninett2010.graphml_uniform_194288830_2.0_0.46" in file_name[4]:
        break
problem = Problem.from_file(file_name[3], file_name[4])
utils.revise_list_commodities(problem)
print(file_name)

output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx, k=num_paths_per_flow,
                                                               number_processors=32,
                                                               base_split=.9, split_type=constants.EXPONENTIAL_DECAY)
path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = output

approx_water_bet_p_mcf_fid_to_flow_rate_mapping, approx_w_bet_p_mcf_dur, _ = approx_water_bet_plus_mcf.get_rates(
    problem,
    path_dict,
    path_edge_idx_mapping,
    path_path_len_mapping,
    path_total_path_len_mapping,
    path_split_ratio_mapping,
    num_paths_per_flow,
    num_iter_approx_water=1,
    num_iter_bet=num_iter_bet,
    link_cap=1000.0,
    break_down=False,
    biased_toward_low_flow_rate=False,
    biased_alpha=0.5,
    link_cap_scale_factor=1000.0,
    min_beta=0,
    num_bins=5,
    min_epsilon=1e-2,
)
alpha = 2
U = 0.1

new_approx_water_fid_to_flow_rate_mapping, new_approx_water_dur, _ = geometric_approx_binning.max_min_approx(alpha, U,
                                                                                                             problem,
                                                                                                             path_dict,
                                                                                                             1000.0,
                                                                                                             max_epsilon=1e-6,
                                                                                                             break_down=False,
                                                                                                             link_cap_scale_multiplier=1000.0)
