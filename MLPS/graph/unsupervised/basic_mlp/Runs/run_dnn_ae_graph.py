import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models"))

from dnn_ae_graph import main
if __name__=='__main__': main()
