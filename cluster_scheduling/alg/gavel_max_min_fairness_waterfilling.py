from datetime import datetime

import cvxpy

from gavel.scheduler.policies import max_min_fairness_water_filling
from gavel.scheduler.policies import max_min_fairness
from gavel.scheduler.policies import gandiva


def get_allocation(job_id_to_gpu_to_thru_mapping, job_scale_factors, job_priority_mapping, cluster_spec, ss_sharing,
                   waterfilling_enabled=True, gandiva_policy=False, seed=1):
    st_time = datetime.now()
    if gandiva_policy:
        scheduler = gandiva.GandivaPolicy(seed=seed)
        job_id_to_cluster_to_alloc_mapping = scheduler.get_allocation(job_id_to_gpu_to_thru_mapping,
                                                                      job_scale_factors,
                                                                      cluster_spec)
    else:
        if ss_sharing:
            if waterfilling_enabled:
                scheduler = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPacking(priority_reweighting_policies=None)
                job_id_to_cluster_to_alloc_mapping = scheduler.get_allocation(job_id_to_gpu_to_thru_mapping, job_scale_factors,
                                                                              job_priority_mapping, cluster_spec, verbose=False)
            else:
                scheduler = max_min_fairness.MaxMinFairnessPolicyWithPacking(solver=cvxpy.GUROBI)
                job_id_to_cluster_to_alloc_mapping = scheduler.get_allocation(job_id_to_gpu_to_thru_mapping, job_scale_factors,
                                                                              job_priority_mapping, cluster_spec)
        else:
            if waterfilling_enabled:
                scheduler = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
                    priority_reweighting_policies=None)
                job_id_to_cluster_to_alloc_mapping = scheduler.get_allocation(job_id_to_gpu_to_thru_mapping, job_scale_factors,
                                                                              job_priority_mapping, cluster_spec, verbose=False)
            else:
                scheduler = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver=cvxpy.GUROBI)
                job_id_to_cluster_to_alloc_mapping = scheduler.get_allocation(job_id_to_gpu_to_thru_mapping, job_scale_factors,
                                                                              job_priority_mapping, cluster_spec)
    dur = (datetime.now() - st_time).total_seconds()
    return job_id_to_cluster_to_alloc_mapping, dur

