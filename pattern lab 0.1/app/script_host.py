from __future__ import annotations

"""Runtime loader for user-editable generator scripts.

The goal is pragmatic: let presets point at a .py file, allow editing it in-app,
and (re)load it safely without restarting.

We cache by absolute path + mtime so edits are picked up immediately.
"""

from pathlib import Path
from typing import Callable, Dict, Tuple, Any, Optional
import importlib.util
import sys


_CACHE: Dict[Tuple[str, float, str], Callable[..., Any]] = {}


def invalidate_cache(path: Optional[Path] = None) -> None:
    """Invalidate cached script(s).

    If path is None, clears everything. Otherwise clears entries for that file.
    """
    global _CACHE
    if path is None:
        _CACHE.clear()
        return
    ap = str(Path(path).resolve())
    _CACHE = {k: v for k, v in _CACHE.items() if k[0] != ap}


def load_script_generator(path: Path, entry: str = "generate") -> Callable[..., Any]:
    """Load a generator function from a python file.

    The script should define a callable named by `entry`.
    That callable is expected to accept: (cfg, **kwargs) OR keyword cfg=...
    and return a list of NoteEvent.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    mtime = path.stat().st_mtime
    key = (str(path), float(mtime), str(entry))
    if key in _CACHE:
        return _CACHE[key]

    # Make a unique module name so reloads don't collide.
    mod_name = f"gm_user_script_{abs(hash((str(path), mtime))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {path}")

    module = importlib.util.module_from_spec(spec)
    # Ensure relative imports inside the script can work if user wants them.
    module.__file__ = str(path)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    fn = getattr(module, entry, None)
    if not callable(fn):
        raise AttributeError(f"Script '{path.name}' has no callable '{entry}'")

    _CACHE[key] = fn
    return fn
