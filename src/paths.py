"""Locating the two external SAM3 repositories.

Neither is vendored here. SAM3-DMS is required; vanilla sam3 is optional
(SAM3-DMS ships its own ``sam3`` package, and all IPCAI 2027 runs used SAM3-DMS
alone):

  * ``sam3``     -- Meta's release (github.com/facebookresearch/sam3)
  * ``SAM3-DMS`` -- the fork (github.com/FudanCVL/SAM3-DMS), which supplies the
                    extended ``build_sam3_video_model(..., use_decoupled_selection=...)``
                    signature every configuration in this study is built through.

Search order for each: the environment variable, then ``third_party/<name>``
next to this repository. Set ``SAM3_DMS_ROOT`` / ``SAM3_ROOT`` to override.

**Import order is load-bearing.** Both repositories expose a module named
``sam3``. If the vanilla one wins, ``build_sam3_video_model`` is imported
without ``use_decoupled_selection`` and every run dies with a ``TypeError``.
``ensure_on_syspath()`` therefore forces SAM3-DMS to ``sys.path[0]``, and must
be called before any ``import sam3``.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SPEC = {
    "SAM3-DMS": ("SAM3_DMS_ROOT", "https://github.com/FudanCVL/SAM3-DMS", "88475d9"),
    "sam3": ("SAM3_ROOT", "https://github.com/facebookresearch/sam3", "5dd401d"),
}


def _resolve(name: str) -> Path:
    env_var, url, commit = _SPEC[name]
    candidates = []
    if os.environ.get(env_var):
        candidates.append(Path(os.environ[env_var]).expanduser())
    candidates.append(REPO_ROOT / "third_party" / name)
    candidates.append(Path("/workspace") / name)  # original RunPod layout

    for path in candidates:
        if path.is_dir():
            return path.resolve()

    raise FileNotFoundError(
        f"Could not find the '{name}' repository. Looked in:\n"
        + "".join(f"  {c}\n" for c in candidates)
        + f"Clone it and retry:\n"
        f"  git clone {url} third_party/{name}\n"
        f"  git -C third_party/{name} checkout {commit}\n"
        f"or point {env_var} at an existing clone."
    )


def ensure_on_syspath() -> None:
    """Put both repositories on ``sys.path``, SAM3-DMS first.

    Idempotent, and safe to call after something else has already imported
    either package -- it re-asserts the ordering rather than assuming it.
    """
    dms = str(_resolve("SAM3-DMS"))
    try:
        base = str(_resolve("sam3"))
    except FileNotFoundError:  # _optional: SAM3-DMS contains the sam3 package
        base = None

    for path in (p for p in (base, dms) if p):  # base first so dms ends up ahead
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def default_checkpoint() -> str:
    """Default SAM3 checkpoint location; override with ``SAM3_CHECKPOINT``."""
    return os.environ.get(
        "SAM3_CHECKPOINT", str(REPO_ROOT / "checkpoints" / "sam3.pt")
    )


def default_data_root(dataset: str) -> str:
    """Default dataset root; override with ``ENDOVIS17_ROOT`` / ``ENDOVIS18_ROOT`` /
    ``CHOLECSEG8K_ROOT``."""
    env_var = {"ev17": "ENDOVIS17_ROOT", "ev18": "ENDOVIS18_ROOT", "cholec": "CHOLECSEG8K_ROOT"}[dataset]
    folder = {"ev17": "endovis_2017", "ev18": "endovis_2018", "cholec": "CholecSeg8k"}[dataset]
    return os.environ.get(env_var, str(REPO_ROOT / "data" / folder))
