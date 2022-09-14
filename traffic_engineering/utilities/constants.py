O_epsilon = 0.0000001
EQUI_DEPTH_0_epsilon = 1e-7
INFINITE_FLOW_RATE = 100000000
EXPONENTIAL_DECAY = 0
EXPONENTIAL_DECAY_LEN = 1
EXPONENTIAL_LEN = 1
RANDOM = 2
UNIFORM = 3
APPROX = "approx({})"
APPROX_MCF = "approx({})_mcf"
APPROX_BET = "approx({})_bet"
APPROX_BET_MCF = "approx({})_bet({})_mcf"
APPROX_BET_MCF_BIASED = "approx({})_bet({})_biased_mcf"
SWAN = "SWAN"
DANNA = "Danna"
NEW_APPROX = "Geometric Binner"
ONE_WATERFILLING = "1-waterfilling"

approach_to_valid_for_run_time = {
    ONE_WATERFILLING: [
        # "model",
        ("computation", False),
        # "extract_rate",
    ],
    APPROX: [
        # "model",
        ("computation", False),
        # "extract_rate",
    ],
    DANNA: [
        # "model",
        ("feasibility_total", False),
        # "feasibility_solver",
        ("mcf_total", False),
        # "mcf_solver",
        # 'extract_rate',
    ],
    SWAN: [
        # 'model',
        ('mcf_total', False),
        # 'mcf_solver',
        # 'freeze_time'
    ],
    NEW_APPROX: [
        # 'model',
        # 'extract_rate',
        ('mcf_solver', False),
    ],
    APPROX_BET: [
        # "model",
        ("computation", False),
        # "extract_rate",
    ],
    APPROX_BET_MCF: [
        # "model_approx_bet_p_mcf",
        # "extract_rate_approx_bet_p_mcf",
        # "model",
        ("computation", False),
        # "extract_rate",
        # "model_equi_depth",
        ("solver_time_equi_depth", True),
        # "extract_rate_equi_depth",
    ]
}
TOPOLOGY_TO_APPROX_BET_MCF_PARAMS = {
    'Uninett2010.graphml': (4, 1e-2, 1e-2, 1, 1000, 1, 10, 0.9),
    'Cogentco.graphml': (15, 1e-2, 1e-2, 1, 1000, 1, 10, 0.9),
    'GtsCe.graphml': (12, 1e-2, 1e-2, 1, 1000, 1, 10, 0.9),
    'UsCarrier.graphml': (7, 1e-2, 1e-2, 1, 1000, 1, 20, 1.0),
    'Colt.graphml': (10, 1e-2, 1e-2, 1, 1000, 1, 10, 0.9),
    'TataNld.graphml': (15, 1e-2, 1e-2, 1, 1000, 1, 20, 1.0),
    # 'Kdl.graphml',
}
