from plain_mlp_redirect import exec_centralized_file, inject_default_cli_arg

inject_default_cli_arg("--adp-mode", "depth_only")
exec_centralized_file(__file__, 'MLPS/image/supervised/dae/Models/dae_multihead_sup_stl_adp_width_to_depth.py')
