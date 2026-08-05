"""Test-session hermeticity.

Force Langfuse tracing OFF before any test module imports config/observability:
the repo .env contains REAL Langfuse keys, and without this every real-module
test run would initialize the OTel exporter and ship test spans to the
production Langfuse project. Set here (conftest loads before test collection)
so config.LANGFUSE_TRACING_ENABLED reads false everywhere in the session.
"""

import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
