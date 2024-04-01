import os

import numpy as np

from utilities import utils
from utilities import job_split_ratios
from utilities import constants


class Problem(object):
    def __init__(self, cluster_spec, job_dict=None, oracle_throughputs=None, throughput_model='gavel', job_mode='single',
                 seed=0, problem_scale_factor=1, space_sharing=False, priority_mapping=None):

        self.cluster_spec = cluster_spec
        self.throughput_model = throughput_model
        self.job_mode = job_mode
        self.seed = seed
        self.problem_scale_factor = problem_scale_factor
        self.job_dict = job_dict
        self.oracle_throughputs = oracle_throughputs
        self.space_sharing = space_sharing
        self.priority_mapping = priority_mapping

        self.sparse_job_list = []
        self.gpu_list = list(sorted(cluster_spec.keys()))
        self.capacity_vector = np.empty(shape=len(cluster_spec))
        self.job_id_to_scale_factor = dict()
        self.job_id_to_initial_split_ratio = dict()
        for gid, gpu_type in enumerate(self.gpu_list):
            self.capacity_vector[gid] = cluster_spec[gpu_type]

        if job_dict is not None:
            assert oracle_throughputs is not None
            for jid, job in job_dict.items():
                throughput_list = np.empty(shape=len(self.gpu_list))
                # assert not jid.is_pair()
                job_type_key = (job.job_type, job.scale_factor)
                for gid, gpu in enumerate(self.gpu_list):
                    thru = oracle_throughputs[gpu][job_type_key]['null']
                    throughput_list[gid] = thru
                self.job_id_to_scale_factor[job.job_id] = job.scale_factor
                prio = job.priority_weight
                if self.priority_mapping is not None:
                    prio = self.priority_mapping[job.job_id]
                self.sparse_job_list.append((jid, (prio, job.scale_factor, throughput_list)))

                split_ratio = job_split_ratios.get_split_ratios_specific_job(throughput_list, 1.1,
                                                                             split_type=constants.EXPONENTIAL_DECAY)
                self.job_id_to_initial_split_ratio[jid] = split_ratio
        else:
            if oracle_throughputs is not None:
                raise Exception
            elif throughput_model == 'gavel':
                self.job_id_to_throughput_mapping = self.GavelModelJobs()
            raise Exception


    def print_stats(self):
        print("=" * 10)
        print(f"cluster_spec: {self.cluster_spec}")
        print(f"throughput_model: {self.throughput_model}")
        print(f"job_mode: {self.job_mode}")
        print(f"num_jobs: {len(self.job_id_to_throughput_mapping)}")

    def get_name(self):
        return f"{self.throughput_model}_{self.job_mode}_{self.seed}_{self.problem_scale_factor}"

    def serialize_as_pickle(self, dir_path):
        file_path = os.path.join(dir_path, '{}_job_scheduling.pkl'.format(self.get_name()))
        utils.write_to_file_as_pickle(self.job_id_to_throughput_mapping, dir_path, file_path)


def get_problem(cluster_spec, throughput_model, job_mode, seed, problem_scale_factor):
    return Problem(cluster_spec, throughput_model, job_mode, seed, problem_scale_factor)
