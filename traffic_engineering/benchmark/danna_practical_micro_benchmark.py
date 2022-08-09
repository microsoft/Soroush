import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from alg import danna_practical_max_min_fair
from utilities import shortest_paths
from utilities import utils
from ncflow.lib.problem import Problem


TM_MODEL_LIST = [
    'uniform',
    'bimodal',
    'gravity',
    'poisson-high-inter',
    'poisson-high-inter',
]
TOPO_NAME_LIST = [
    'Uninett2010.graphml',
    # 'Cogentco.graphml',
    # 'GtsCe.graphml',
    'UsCarrier.graphml',
    'Colt.graphml',
    # 'TataNld.graphml',
    # 'Kdl.graphml',
]

# TM_MODEL_LIST = ['gravity']
# TOPO_NAME_LIST = ['UsCarrier.graphml', 'Colt.graphml']
feasibility_grb_method = [1]
mcf_grb_method = [0, 1, 2]
num_path_list = [16]
# num_path_list = [4]
link_cap = 1000.0
log_dir = "../outputs"
fid = utils.get_fid()
log_file = f"../outputs/danna_practical_{fid}.txt"
log_folder_flows = f"../outputs/danna_practical_{fid}/"
U = 0.1
num_scenario_per_topo_traffic = 2

for num_paths in num_path_list:
    for topo_name in TOPO_NAME_LIST:
        for tm_model in TM_MODEL_LIST:
            fnames = np.random.choice(utils.find_topo_tm_fname(topo_name, tm_model), num_scenario_per_topo_traffic)
            example_file = fnames[0]
            problem = Problem.from_file(example_file[3], example_file[4])
            path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                                k=num_paths, number_processors=32)
            paths, path_edge_idx_mapping, _, _, _ = path_output

            for file_name in fnames:
                for feasibility_method in feasibility_grb_method:
                    for mcf_method in mcf_grb_method:
                        problem = Problem.from_file(file_name[3], file_name[4])
                        utils.revise_list_commodities(problem)
                        per_flow_log_file_name = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths}"
                        print("=" * 5, file_name, per_flow_log_file_name, "=" * 5)
                        danna_output = danna_practical_max_min_fair.get_rates(U, problem, paths, link_cap,
                                                                              feasibility_grb_method=feasibility_method,
                                                                              mcf_grb_method=mcf_method,
                                                                              break_down=False)
                        danna_fid_to_total_rate, danna_fid_to_per_path_rate, danna_dur, danna_run_time_dict = danna_output

                        danna_total_flow = 0
                        for (src, dst) in danna_fid_to_total_rate:
                            danna_total_flow += danna_fid_to_total_rate[src, dst]

                        log_txt = f"{feasibility_method},{mcf_method},{file_name},{num_paths},{danna_total_flow},{danna_dur},{per_flow_log_file_name}\n"
                        utils.write_to_file(log_txt, log_dir, log_file)
                        print("result:", log_txt)
                        for key, time_s in danna_run_time_dict.items():
                            log_txt = f"    {key}: {time_s} s"
                            utils.write_to_file(log_txt, log_dir, log_file)
                            print("      ", log_txt)
