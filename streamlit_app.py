"""Stable Streamlit entry point outside the package's helper ``pages`` directory."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bankruptcy_prediction.ui.app import main


if __name__ == "__main__":
    main()
