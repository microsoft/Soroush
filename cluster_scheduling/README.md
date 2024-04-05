# Cluster Scheduling
This folder contains the implementation of `Soroush` for the cluster scheduling use case. Our implementation uses CVXPY v1.2.1 for modeling and the Gurobi optimization solver as the backend. 

## Instructions
Follow these guidelines to compare Soroush with Gavel:
-  Download Gavel from https://github.com/stanford-futuredata/gavel.git
-  We used Gurobi in our evaluations. To reproduce our results, please change Gavel's solver to point to Gurobi. Specifically, change the following two lines in `scheduler/policies/max_min_fairness_water_filling.py`
    - change line 150 to: 
    ```bash
    result = self._lp.solve(solver=cp.GUROBI, warm_start=True)
    ```
    
    - change line 229 to:

    ```bash
    result = self._milp.solve(solver=cp.GUROBI, warm_start=True)
    ```
