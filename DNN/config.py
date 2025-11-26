VALID_CASES = [
    'pglib_opf_case14_ieee',
    'pglib_opf_case30_ieee', 
    'pglib_opf_case57_ieee', 
    'pglib_opf_case118_ieee', 
    'pglib_opf_case500_goc', 
    'pglib_opf_case2000_goc', 
    'pglib_opf_case6470_rte', 
    'pglib_opf_case4661_sdetpglib_opf_case10000_goc', 
    'pglib_opf_case13659_pegase'
    # Extend this list as needed
]

# initial learning rate for Adam
OPTIMIZER_LR = 1e-3

SCHEDULER_PARAMS = {"T_0": 10, "T_mult": 1, "eta_min": 0.0}
