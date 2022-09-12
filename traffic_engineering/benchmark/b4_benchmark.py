import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from alg import b4_experience_w_global_software_wan as b4
from alg import b4_extended as b4_plus
from utilities import shortest_paths
from utilities import utils
from ncflow.lib.problem import Problem


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
    # 'Colt.graphml',
    'TataNld.graphml',
    # 'Kdl.graphml',
]

feasibility_grb_method = 1
mcf_grb_method = 1
num_path_list = [16]
link_cap = 1000.0
use_b4_plus = True
b4_class = b4
if use_b4_plus:
    b4_class = b4_plus
log_dir = "../outputs"
fid = utils.get_fid()
log_file = f"../outputs/b4_{fid}_{use_b4_plus}.txt"
log_folder_flows = f"../outputs/b4_{fid}_{use_b4_plus}/"
U = 0

for num_paths in num_path_list:
    for topo_name in TOPO_NAME_LIST:
        for tm_model in TM_MODEL_LIST:
            fnames = utils.find_topo_tm_fname(topo_name, tm_model)
            example_file = fnames[0]
            problem = Problem.from_file(example_file[3], example_file[4])
            path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                                k=num_paths, number_processors=32)
            path_dict, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping = path_output

            for file_name in fnames:
                problem = Problem.from_file(file_name[3], file_name[4])
                utils.revise_list_commodities(problem)
                per_flow_log_file_name = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths}"
                print("=" * 5, file_name, per_flow_log_file_name, "=" * 5)
                b4_output = b4_class.get_rates(problem, num_paths, link_cap,
                                               path_edge_idx_mapping=path_edge_idx_mapping,
                                               ath_path_len_mapping=path_path_len_mapping)
                b4_fid_to_per_path_rate, b4_dur, b4_run_time_dict = b4_output

                b4_total_flow = 0
                for (src, dst) in b4_fid_to_per_path_rate:
                    b4_total_flow += np.sum(b4_fid_to_per_path_rate[src, dst])

                per_flow_log_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_per_path_flow.pickle"
                utils.write_to_file_as_pickle(b4_fid_to_per_path_rate, log_folder_flows, per_flow_log_file_path)

                run_time_dict_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_run_time_dict.pickle"
                utils.write_to_file_as_pickle(b4_run_time_dict, log_folder_flows, run_time_dict_file_path)

                log_txt = f"{file_name},{num_paths},{b4_total_flow},{b4_dur},{per_flow_log_file_name}\n"
                utils.write_to_file(log_txt, log_dir, log_file)
                print("result:", log_txt)
