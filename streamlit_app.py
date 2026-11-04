"""Stable entrypoint: lightweight research snapshot by default.

Streamlit Community Cloud runs this file without local-only environment flags,
so the public URL serves the static research snapshot. The Windows launcher
can still opt into the original API-backed application for local development.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _run_full_local_app() -> None:
    src_root = PROJECT_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from bankruptcy_prediction.ui.app import main

    main()


def _run_research_snapshot() -> None:
    from deploy.research_app import main

    main()


if __name__ == "__main__":
    if os.environ.get("BANKRUPTCY_FULL_LOCAL_APP") == "1":
        _run_full_local_app()
    else:
        _run_research_snapshot()
