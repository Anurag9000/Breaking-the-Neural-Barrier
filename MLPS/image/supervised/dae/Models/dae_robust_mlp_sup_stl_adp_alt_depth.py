from plain_mlp_redirect import exec_centralized_file, inject_default_cli_arg

inject_default_cli_arg("--adp-mode", "alt_depth")
exec_centralized_file(__file__, 'MLPS/image/supervised/dae/Models/dae_robust_mlp_sup_stl_adp_width_to_depth.py')
