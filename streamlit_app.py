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
    configured_api = os.environ.get("BANKRUPTCY_API_BASE_URL")
    embedded_reload = os.environ.get("BANKRUPTCY_EMBEDDED_API") == "1"
    legacy_cloud_api = configured_api in {
        "http://127.0.0.1:8765",
        "http://127.0.0.1:8766",
    }
    if not configured_api or embedded_reload or legacy_cloud_api:
        # Keep the cloud-only service away from the legacy port 8000.  A
        # Streamlit hot reload can leave the previous daemon thread alive;
        # using a deployment-specific port guarantees that the UI talks to
        # the API created from the current source revision.
        cloud_api_port = 8767
        os.environ["BANKRUPTCY_EMBEDDED_API"] = "1"
        os.environ["BANKRUPTCY_API_BASE_URL"] = (
            f"http://127.0.0.1:{cloud_api_port}"
        )
        ensure_embedded_api(port=cloud_api_port)
    main()
