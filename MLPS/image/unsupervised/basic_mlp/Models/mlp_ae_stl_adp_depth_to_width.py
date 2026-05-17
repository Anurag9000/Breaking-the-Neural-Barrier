from plain_mlp_redirect import exec_centralized_file, inject_default_cli_arg

inject_default_cli_arg("--adp-mode", "depth_to_width")
exec_centralized_file(__file__, 'MLPS/image/unsupervised/basic_mlp/Models/mlp_ae_stl_adp_width_to_depth.py')
