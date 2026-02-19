import signal
from dotenv import load_dotenv

from prokaryotes.imap_v1 import IngestController
from prokaryotes.utils import setup_logging

load_dotenv()
setup_logging()

controller = IngestController()
signal.signal(signal.SIGINT, controller.graceful_shutdown)
signal.signal(signal.SIGTERM, controller.graceful_shutdown)
controller.run()
