import json
from pathlib import Path

request = json.loads(Path("change_request.json").read_text())
config_src = Path("app_config.py").read_text()
notes = Path("notes.md").read_text()

namespace = {}
exec(config_src, namespace)

assert namespace["SERVICE_NAME"] == request["service_name"]
assert namespace["TIMEOUT_SECONDS"] == request["timeout_seconds"]
assert namespace["MAX_RETRIES"] == request["max_retries"]
assert namespace["ENABLE_CACHE"] is request["enable_cache"]
assert "LEGACY_MODE" not in namespace
assert namespace["build_banner"]() == "Atlas timeout=45 retries=4 cache=True"

expected_notes = (
    "# Atlas Service\n\n"
    "## Owners\n"
    "- Mina\n"
    "- Ravi\n\n"
    "## Summary\n"
    "Service: Atlas\n"
    "Timeout: 45\n"
    "Retries: 4\n"
    "Cache: enabled\n\n"
    "## Next Steps\n"
    "- add cache metrics\n"
    "- publish retry guidance\n"
)
assert notes == expected_notes, notes
print("PASS")
