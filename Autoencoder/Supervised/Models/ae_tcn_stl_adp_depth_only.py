import sys
from Autoencoder.Supervised.Models.ae_tcn_stl_adp_width_to_depth import main

if __name__ == "__main__":
    if "--adp-mode" not in sys.argv:
        sys.argv += ["--adp-mode", "depth_only"]
    main()
