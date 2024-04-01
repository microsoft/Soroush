# Cluster Scheduling
### Instructions
1. Download Gavel from https://github.com/stanford-futuredata/gavel.git
2. In our evaluations, we used Gurobi, so please change the Gavel's solver to point to Gurobi. Specifically, change the solvers in line 150 and 229 of 'scheduler/policies/max_min_fairness_water_filling.py' to cp.GUROBI.

