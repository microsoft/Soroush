import os
import multiprocessing

import numpy as np

from itertools import product

from cluster_scheduling.utilities import utils
from cluster_scheduling.scripts import problem


THROUGHPUT_MODELS = ['gavel']
JOB_MODES = ['single_worker']
CLUSTER_GPU_TYPES = ['gavel']
NUM_GPUS_MODEL = ['uniform']
SCALE_FACTORS = [1., 2., 4., 8., 16., 32., 64., 128.]


PROBLEM_DIR = "./generated_problems/"
NUM_SAMPLES = 5


def create_problem(args):
    throughput_mode, job_mode, problem_scale_factor, cluster_spec = args
    thru_model_dir = os.path.join(PROBLEM_DIR, throughput_mode)
    for _ in range(NUM_SAMPLES):
        curr_problem = problem.get_problem(cluster_spec, throughput_mode, job_mode,
                                           seed=np.random.randint(2 ** 31 - 1),
                                           problem_scale_factor=problem_scale_factor)
        curr_problem.print_stats()
        curr_problem.serialize_as_pickle(thru_model_dir)


if __name__ == "__main__":
    utils.ensure_dir(PROBLEM_DIR)
    for throughput_model in THROUGHPUT_MODELS:
        thru_model_dir = os.path.join(PROBLEM_DIR, throughput_model)
        utils.ensure_dir(thru_model_dir)

    pool = multiprocessing.Pool(10)
    pool.map(create_problem, product(THROUGHPUT_MODELS, JOB_MODES, SCALE_FACTORS, CLUSTER_GPU_TYPES, NUM_GPUS_MODEL))
