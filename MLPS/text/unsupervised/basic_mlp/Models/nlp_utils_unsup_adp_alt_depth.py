from plain_mlp_redirect import exec_centralized_file, inject_default_cli_arg

inject_default_cli_arg("--adp-mode", "alt_depth")
exec_centralized_file(__file__, 'MLPS/text/unsupervised/basic_mlp/Models/nlp_utils_unsup_adp_width_to_depth.py')
