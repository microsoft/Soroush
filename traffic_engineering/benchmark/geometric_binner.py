import sys
import os

# sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from alg import geometric_approx_binning
from utilities import shortest_paths
from utilities import utils
from ncflow.lib.problem import Problem


# TM_MODEL_LIST = ['bimodal', 'gravity']
# TOPO_NAME_LIST = ['Uninett2010.graphml', 'Cogentco.graphml', 'GtsCe.graphml']

# TM_MODEL_LIST = ['uniform'] #'poisson-high-inter'] #'poisson-high-intra']
# TOPO_NAME_LIST = ['Uninett2010.graphml', 'Cogentco.graphml', 'GtsCe.graphml', 'UsCarrier.graphml',
#                   'Colt.graphml']

# TM_MODEL_LIST = ['bimodal', 'gravity']
# TOPO_NAME_LIST = ['UsCarrier.graphml', 'Colt.graphml']

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

# num_path_list = [4, 16]
mcf_grb_method = 2
num_path_list = [16]
link_cap = 1000.0
log_dir = "../outputs"
fid = utils.get_fid()
log_file = f"../outputs/geometric_binner_{fid}.txt"
log_folder_flows = f"../outputs/geometric_binner_{fid}/"
alpha = 2
U = 0.1
max_epsilon = 1e-6
link_cap_multiplier = 1

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

                geometric_approx_binner_output = geometric_approx_binning.max_min_approx(alpha, U, problem,
                                                                                         path_dict, link_cap,
                                                                                         max_epsilon=max_epsilon,
                                                                                         link_cap_scale_multiplier=link_cap_multiplier,
                                                                                         mcf_grb_method=mcf_grb_method)

                gb_fid_to_flow_rate_mapping, gb_duration, gb_run_time_dict = geometric_approx_binner_output

                gb_total_flow = 0
                for (src, dst) in gb_fid_to_flow_rate_mapping:
                    gb_total_flow += sum(gb_fid_to_flow_rate_mapping[src, dst])

                per_flow_log_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_per_path_flow.pickle"
                utils.write_to_file_as_pickle(gb_fid_to_flow_rate_mapping, log_folder_flows, per_flow_log_file_path)

                run_time_dict_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_run_time_dict.pickle"
                utils.write_to_file_as_pickle(gb_run_time_dict, log_folder_flows, run_time_dict_file_path)

                log_txt = f"{file_name},{num_paths},{gb_total_flow},{gb_duration},{per_flow_log_file_name}\n"
                utils.write_to_file(log_txt, log_dir, log_file)
                print("result:", log_txt)
