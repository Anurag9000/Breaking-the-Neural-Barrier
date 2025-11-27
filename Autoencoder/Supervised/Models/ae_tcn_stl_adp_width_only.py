# ADP WRAPPER
# - All ADP algorithms are implemented in Autoencoder.Supervised.Models.ae_tcn_stl_adp_width_to_depth.
# - This file only selects the appropriate mode and delegates to the core.
import sys
from Autoencoder.Supervised.Models.ae_tcn_stl_adp_width_to_depth import main

if __name__ == "__main__":
    if "--adp-mode" not in sys.argv:
        sys.argv += ["--adp-mode", "width_only"]
    main()
