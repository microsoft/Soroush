import sys
import os

# sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from alg import swan_max_min_approx
from utilities import shortest_paths
from utilities import utils
from ncflow.lib.problem import Problem


# TM_MODEL_LIST = ['bimodal', 'gravity']
# TOPO_NAME_LIST = ['Uninett2010.graphml', 'Cogentco.graphml', 'GtsCe.graphml']

# TM_MODEL_LIST = ['uniform'] #'poisson-high-inter'] #'poisson-high-intra']
# TOPO_NAME_LIST = ['Uninett2010.graphml', 'Cogentco.graphml', 'GtsCe.graphml', 'UsCarrier.graphml',
#                   'Colt.graphml']

TM_MODEL_LIST = [
    'uniform',
    'bimodal',
    'gravity',
    'poisson-high-inter',
    'poisson-high-intra',
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

# num_path_list = [4, 16]
mcf_grb_method = 1
num_path_list = [16]
# num_path_list = [4]
link_cap = 1000.0
log_dir = "../outputs"
fid = utils.get_fid()
log_file = f"../outputs/swan_{fid}.txt"
log_folder_flows = f"../outputs/swan_{fid}/"
alpha = 2
U = 0.1

# only_consider = [162, 175, 181, 191, 197, 402, 415, 421, 431, 437]
only_consider = None
skip_less_than = -1
current = 1
for num_paths in num_path_list:
    for topo_name in TOPO_NAME_LIST:
        for tm_model in TM_MODEL_LIST:
            fnames = utils.find_topo_tm_fname(topo_name, tm_model)
            example_file = fnames[0]
            problem = Problem.from_file(example_file[3], example_file[4])
            path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                                k=num_paths, number_processors=32)
            path_dict, path_edge_idx_mapping, _, _, _ = path_output
            for file_name in fnames:
                if only_consider is not None and current not in only_consider:
                    current += 1
                    continue
                elif current < skip_less_than:
                    current += 1
                    continue
                current += 1
                problem = Problem.from_file(file_name[3], file_name[4])
                utils.revise_list_commodities(problem)
                per_flow_log_file_name = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths}"
                print("=" * 5, file_name, per_flow_log_file_name, "=" * 5)

                swan_output = swan_max_min_approx.max_min_approx(alpha, U, problem, path_dict, link_cap,
                                                                 mcf_grb_method=mcf_grb_method,
                                                                 break_down=False)
                swan_fid_to_flow_rate_mapping, swan_dur, swan_run_time_dict = swan_output

                swan_total_flow = 0
                for (src, dst) in swan_fid_to_flow_rate_mapping:
                    swan_total_flow += sum(swan_fid_to_flow_rate_mapping[src, dst])

                per_flow_log_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_per_path_flow.pickle"
                utils.write_to_file_as_pickle(swan_fid_to_flow_rate_mapping, log_folder_flows, per_flow_log_file_path)

                run_time_dict_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_run_time_dict.pickle"
                utils.write_to_file_as_pickle(swan_run_time_dict, log_folder_flows, run_time_dict_file_path)

                log_txt = f"{file_name},{num_paths},{swan_total_flow},{swan_dur},{per_flow_log_file_name}\n"
                utils.write_to_file(log_txt, log_dir, log_file)
                print("result:", log_txt)
