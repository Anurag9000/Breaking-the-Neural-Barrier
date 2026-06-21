import importlib.util
from pathlib import Path


BASE_PATH = Path(__file__).with_name("nlp_ae_common_adp_width_to_depth.py").resolve()
_spec = importlib.util.spec_from_file_location("adp_impl", BASE_PATH)
adp_impl = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(adp_impl)


ADPConfig = adp_impl.ADPConfig  # type: ignore
adp_search = adp_impl.adp_search  # type: ignore


def main():
    import subprocess
    import sys

    subprocess.call([sys.executable, str(BASE_PATH)])


if __name__ == "__main__":
    try:
        import os as _os, sys as _sys
        if _os.name == "posix" and _sys.platform.startswith("linux"):
            import ctypes as _ctypes
            _ctypes.CDLL("libc.so.6", use_errno=True).mlockall(3)
        elif _os.name == "nt":
            import ctypes as _ctypes
            _ctypes.windll.kernel32.SetProcessWorkingSetSize(_ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
    except Exception:
        pass
    main()
