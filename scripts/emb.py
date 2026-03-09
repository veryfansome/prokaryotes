from dotenv import load_dotenv

from prokaryotes.emb_v1 import EmbeddingV1
from prokaryotes.utils import setup_logging

load_dotenv()
setup_logging()

v1 = EmbeddingV1("Snowflake/snowflake-arctic-embed-l-v2.0")
