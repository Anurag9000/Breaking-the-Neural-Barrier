"""Alias runner for multi-head DAE classifier.

Reuses the group-sparse MLP DAE supervised runner.
"""

from .run_dae_groupsparse_mlp_sup_stl import main as _base_main


def main() -> None:
    _base_main()


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

