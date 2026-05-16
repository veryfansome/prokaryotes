SERVICE_NAME = "Beacon"
TIMEOUT_SECONDS = 15
ENABLE_CACHE = False
LEGACY_MODE = True

def build_banner():
    return f"{SERVICE_NAME} timeout={TIMEOUT_SECONDS} cache={ENABLE_CACHE}"
