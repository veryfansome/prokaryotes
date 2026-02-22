import os
from dotenv import load_dotenv

from prokaryotes.imap_v1 import IngestController
from prokaryotes.utils import setup_logging

load_dotenv()
setup_logging()

imap_host = os.getenv('IMAP_HOST')
imap_username = os.getenv('IMAP_USERNAME')
imap_password = os.getenv('IMAP_PASSWORD')
controller = IngestController("INBOX", imap_host, imap_username, imap_password, max_workers=1)
controller.run()
