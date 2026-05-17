from plain_mlp_redirect import exec_centralized_file, inject_default_cli_arg

inject_default_cli_arg("--adp-mode", "alt_width")
exec_centralized_file(__file__, 'MLPS/text/unsupervised/basic_mlp/Models/nlp_utils_ssl_adp_width_to_depth.py')
