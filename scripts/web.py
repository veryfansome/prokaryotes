from dotenv import load_dotenv

from prokaryotes.utils import setup_logging
from prokaryotes.web_v1 import ProkaryoteV1

load_dotenv()
setup_logging()

v1 = ProkaryoteV1("scripts")
