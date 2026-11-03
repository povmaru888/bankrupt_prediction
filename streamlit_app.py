"""Stable Streamlit entry point outside the package's helper ``pages`` directory."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bankruptcy_prediction.ui.app import main  # noqa: E402
from bankruptcy_prediction.ui.embedded_api import ensure_embedded_api  # noqa: E402

if __name__ == "__main__":
    # The local launcher supplies BANKRUPTCY_API_BASE_URL and keeps the API in
    # its own process.  Streamlit Community Cloud starts only this script, so
    # it receives a loopback-only API thread inside the same container.
    if not os.environ.get("BANKRUPTCY_API_BASE_URL"):
        ensure_embedded_api()
    main()
