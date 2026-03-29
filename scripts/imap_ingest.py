import logging
import os

from dotenv import load_dotenv

from prokaryotes.imap_v1 import IngestController
from prokaryotes.utils_v1.logging_utils import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()
setup_logging()

# TODO: Change this to something that would be viable for multiple users
imap_host = os.getenv('IMAP_HOST')
imap_username = os.getenv('IMAP_USERNAME')
imap_password = os.getenv('IMAP_PASSWORD')
if imap_host and imap_username and imap_password:
    controller = IngestController("INBOX", imap_host, imap_username, imap_password, max_workers=1)
    controller.run()
else:
    logger.info("IMAP credentials no found")
