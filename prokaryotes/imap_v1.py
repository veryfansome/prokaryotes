import asyncio
import certifi
import logging
import os
import ssl
from email import policy
from email.parser import BytesParser
from imapclient import IMAPClient
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EmailMessage(BaseModel):
    id: int
    subject: str
    sender: str
    timestamp: str
    body: str

class EmailReader:
    def __init__(self, imap_host: str, imap_username: str, imap_password: str):
        self.imap_host = imap_host
        self.imap_password = imap_password
        self.imap_username = imap_username

    @property
    def imap_client(self):
        imap_client = IMAPClient(
            self.imap_host, ssl=True, ssl_context=ssl.create_default_context(cafile=certifi.where())
        )
        imap_client.use_uid = True
        imap_client.login(self.imap_username, self.imap_password)
        return imap_client

    def fetch_unread(self, limit: int = 1) -> list[EmailMessage]:
        fetched_messages = []
        with self.imap_client as server:
            logger.info(f"Folders: {server.list_folders()}")
            server.select_folder("INBOX")
            unseen_messages = server.search(["UNSEEN"])
            unseen_messages_len = len(unseen_messages)
            logger.info(f"Found {unseen_messages_len} unseen messages")
            if unseen_messages_len < limit:
                return fetched_messages

            response = server.fetch(unseen_messages[:limit], ["BODY.PEEK[]"])
            for msg_id, data in response.items():
                raw_email = data.get(b"BODY[]")
                if raw_email is None:
                    raise KeyError(f"Expected BODY[] but got keys: {list(data.keys())}")
                msg = BytesParser(policy=policy.default).parsebytes(raw_email)
                fetched_messages.append(EmailMessage(
                    id=msg_id,
                    subject=msg["subject"],
                    sender=msg["from"],
                    timestamp=msg["date"],
                    body=self.extract_text_from_message(msg),
                ))
            return fetched_messages

    @classmethod
    def extract_text_from_message(cls, msg):
        """Extract plain text content from a parsed email.message.EmailMessage."""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    return part.get_content()
        else:
            return msg.get_content()
        return None

async def main(email_reader: EmailReader):
    messages = await asyncio.to_thread(email_reader.fetch_unread)
    logger.info(messages)

if __name__ == "__main__":
    from dotenv import load_dotenv
    from prokaryotes.utils import setup_logging

    load_dotenv()
    setup_logging()

    asyncio.run(main(EmailReader(
        os.environ.get('IMAP_HOST'), os.environ.get('IMAP_USERNAME'), os.environ.get('IMAP_PASSWORD')
    )))
