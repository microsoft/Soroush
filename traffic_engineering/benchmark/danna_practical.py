import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from alg import danna_practical_max_min_fair
from utilities import shortest_paths
from utilities import utils
from ncflow.lib.problem import Problem

# TM_MODEL = 'bimodal'
# TOPO_NAME = 'Kdl.graphml'
# TOPO_NAME = 'Cogentco.graphml'
# TOPO_NAME = 'Ion.graphml'
# TOPO_NAME = 'Uninett2010.graphml'
# fnames = utils.find_topo_tm_fname(TOPO_NAME, TM_MODEL)


# Note (@Pooria):
# 601-th TM ('GtsCe.graphml', 'poisson-high-inter', fid=0): gurobi barrier method faces segmentation fault:
# should change gurobi method = 1 (dual simplex)

# done combinations
# ['Uninett2010.graphml', 'Cogentco.graphml'] * ['uniform', 'bimodal', 'gravity', 'poisson-high-inter'] * [4, 16]
# ['GtsCe.graphml'] * ['uniform', 'bimodal', 'gravity'] * [4, 16]
# ['UsCarrier.graphml', 'Colt.graphml'] * ['uniform'] * [4, 16]
# ['UsCarrier.graphml', 'Colt.graphml'] * ['bimodal'] * [4]
# ['UsCarrier.graphml', 'Colt.graphml'] * ['gravity'] * [4]
# ['GtsCe.graphml', 'UsCarrier.graphml', 'Colt.graphml'] * ['poisson-high-inter'] * [4]

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

# TM_MODEL_LIST = ['gravity']
# TOPO_NAME_LIST = ['UsCarrier.graphml', 'Colt.graphml']
feasibility_grb_method = 1
mcf_grb_method = 1
num_path_list = [16]
# num_path_list = [4]
link_cap = 1000.0
log_dir = "../outputs"
fid = utils.get_fid()
log_file = f"../outputs/danna_practical_{fid}.txt"
log_folder_flows = f"../outputs/danna_practical_{fid}/"
U = 0
skip_until = -1
curr = 1

for num_paths in num_path_list:
    for topo_name in TOPO_NAME_LIST:
        for tm_model in TM_MODEL_LIST:
            fnames = utils.find_topo_tm_fname(topo_name, tm_model)
            example_file = fnames[0]
            problem = Problem.from_file(example_file[3], example_file[4])
            path_output = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                                k=num_paths, number_processors=32)
            paths, path_edge_idx_mapping, _, _, _ = path_output

            for file_name in fnames:
                if curr < skip_until:
                    print(f"skipping {curr} {file_name}")
                    curr += 1
                    continue
                problem = Problem.from_file(file_name[3], file_name[4])
                utils.revise_list_commodities(problem)
                per_flow_log_file_name = file_name[4].split("/")[-1][:-4] + f"_num_paths_{num_paths}"
                print("=" * 5, file_name, per_flow_log_file_name, "=" * 5)
                danna_output = danna_practical_max_min_fair.get_rates(U, problem, paths, link_cap,
                                                                      feasibility_grb_method=feasibility_grb_method,
                                                                      mcf_grb_method=mcf_grb_method,
                                                                      break_down=False)
                danna_fid_to_total_rate, danna_fid_to_per_path_rate, danna_dur, danna_run_time_dict = danna_output

                danna_total_flow = 0
                for (src, dst) in danna_fid_to_total_rate:
                    danna_total_flow += danna_fid_to_total_rate[src, dst]

                total_flow_log_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_total_flow.pickle"
                utils.write_to_file_as_pickle(danna_fid_to_total_rate, log_folder_flows, total_flow_log_file_path)

                per_flow_log_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_per_path_flow.pickle"
                utils.write_to_file_as_pickle(danna_fid_to_per_path_rate, log_folder_flows, per_flow_log_file_path)

                run_time_dict_file_path = log_folder_flows + "/" + per_flow_log_file_name + "_run_time_dict.pickle"
                utils.write_to_file_as_pickle(danna_run_time_dict, log_folder_flows, run_time_dict_file_path)

                log_txt = f"{file_name},{num_paths},{danna_total_flow},{danna_dur},{per_flow_log_file_name}\n"
                utils.write_to_file(log_txt, log_dir, log_file)
                print("result:", log_txt)
