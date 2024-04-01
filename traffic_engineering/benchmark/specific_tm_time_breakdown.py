import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from alg import approx_waterfilling
from utilities import shortest_paths
from alg import swan_max_min_approx
from utilities import utils
from ncflow.lib.problem import Problem
from alg import k_waterfilling


TM_MODEL = 'bimodal'
TOPO_NAME = 'Cogentco.graphml'


fnames = utils.find_topo_tm_fname(TOPO_NAME, TM_MODEL)
num_paths_per_flow = 16

file_name = fnames[4]
problem = Problem.from_file(file_name[3], file_name[4])

path_dict, path_edge_idx_mapping = shortest_paths.all_pair_shortest_path_multi_processor(problem.G, problem.edge_idx,
                                                                  k=num_paths_per_flow, number_processors=32)

approx_water_fid_to_flow_rate_mapping, approx_dur = approx_waterfilling.get_rates(problem, path_edge_idx_mapping,
                                                                                  num_paths_per_flow, num_iter=1,
                                                                                  link_cap=1000.0, break_down=True)

k_water_fid_to_flow_rate_mapping, water_dur = k_waterfilling.get_max_min_fair_multi_path_constrained_k_waterfilling(problem,
                                                                                                                    path_edge_idx_mapping,
                                                                                                                    num_paths_per_flow=num_paths_per_flow,
                                                                                                                    link_cap=1000.0,
                                                                                                                    k=1,
                                                                                                                    break_down=True)

alpha = 2
U = 0.1
swan_fid_to_flow_rate_mapping, swan_dur = swan_max_min_approx.max_min_approx(alpha, U, problem, path_dict, 1000, break_down=True)

print(file_name)
print(approx_dur)
print(water_dur)
print(swan_dur)