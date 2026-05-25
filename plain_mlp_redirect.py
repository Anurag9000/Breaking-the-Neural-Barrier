from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional


REPO_ROOT = Path(__file__).resolve().parent
SUPPORTED_ADP_MODES = {"alt_width", "width_to_depth"}


def _infer_package_name(source_path: Path) -> Optional[str]:
    try:
        relative = source_path.relative_to(REPO_ROOT)
    except ValueError:
        return None

    package_parts = list(relative.parent.parts)
    if not package_parts:
        return None
    return ".".join(package_parts)


def exec_centralized_file(source_file: str, relative_target: str, extra_globals: Optional[Dict[str, object]] = None) -> None:
    source_path = Path(source_file).resolve()
    target = REPO_ROOT / relative_target
    if not target.exists():
        raise ImportError(f"Centralized MLP file not found: {target}")

    caller_globals = sys._getframe(1).f_globals
    caller_globals["__file__"] = str(target)
    caller_globals.setdefault("__cached__", None)
    caller_globals.setdefault("__loader__", None)
    if not caller_globals.get("__package__"):
        inferred_package = _infer_package_name(source_path)
        if inferred_package:
            caller_globals["__package__"] = inferred_package
    if extra_globals:
        caller_globals.update(extra_globals)

    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    parent = str(target.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    code = compile(target.read_text(encoding="utf-8-sig"), str(target), "exec")
    exec(code, caller_globals)


def inject_default_cli_arg(flag: str, value: str) -> None:
    if __name__ != "__main__":
        return
    if flag == "--adp-mode" and value not in SUPPORTED_ADP_MODES:
        supported = ", ".join(sorted(SUPPORTED_ADP_MODES))
        raise SystemExit(
            f"ADP mode '{value}' is disabled by repo policy. Supported ADP modes are: {supported}."
        )
    argv = sys.argv[1:]
    if flag in argv:
        return
    sys.argv[1:1] = [flag, value]
