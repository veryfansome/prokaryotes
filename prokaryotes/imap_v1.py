import base64
import certifi
import logging
import os
import quopri
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.parser import HeaderParser
from email.utils import parsedate_to_datetime
from imapclient import IMAPClient

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Folder:
    flags: list[str]
    delimiter: str
    name: str

@dataclass(frozen=True)
class MessagePartRef:
    folder: str
    uid: int
    section: str
    content_type: str
    encoding: str | None
    filename: str | None
    size: int | None
    charset: str | None

@dataclass(frozen=True)
class Message:
    folder: str
    uid: int
    message_id: str
    subject: str
    sender: str
    timestamp: datetime
    body: MessagePartRef | None
    calendar_invites: list[MessagePartRef]
    attachments: list[MessagePartRef]
    in_reply_to: str | None = None
    references: list[str] = None

class EmailReader:
    def __init__(self, imap_host: str, imap_username: str, imap_password: str):
        self.imap_host = imap_host
        self.imap_password = imap_password
        self.imap_username = imap_username

    def fetch_messages(self, folder="INBOX", criteria="ALL", limit: int = 1) -> list[Message]:
        fetched_messages = []
        try:
            with self.imap_client as server:
                server.select_folder(folder)
                unseen_messages = server.search([criteria])
                unseen_messages_len = len(unseen_messages)
                logger.info(f"Found {unseen_messages_len} unseen messages")
                if unseen_messages_len == 0:
                    return fetched_messages

                header_fields = b"HEADER.FIELDS (SUBJECT FROM DATE MESSAGE-ID IN-REPLY-TO REFERENCES)"
                body_structure_field = b"BODYSTRUCTURE"
                response = server.fetch(unseen_messages[:limit], [
                    b"BODY.PEEK[%s]" % header_fields,
                    body_structure_field
                ])
                for uid, body_data in response.items():
                    headers = parse_header(bytes2str(body_data[b"BODY[%s]" % header_fields]))
                    logger.debug((uid, headers, body_data[body_structure_field]))
                    try:
                        (
                            html_parts,
                            plain_parts,
                            calendar_parts,
                            attachment_parts
                        ) = classify_parts(
                            folder, uid, body_data[body_structure_field],
                        )
                        body_part = choose_largest_part(html_parts) if html_parts else (
                            choose_largest_part(plain_parts) if plain_parts else None
                        )
                        msg = Message(
                            folder=folder,
                            uid=uid,
                            message_id=headers["message-id"],
                            subject=headers["subject"],
                            sender=headers["from"],
                            timestamp=parsedate_to_datetime(headers["date"]),
                            body=body_part,
                            calendar_invites=calendar_parts,
                            attachments=attachment_parts,
                            in_reply_to=headers.get("in-reply-to"),
                            references=parse_references(headers.get("references")),
                        )
                        logger.debug(msg)
                        fetched_messages.append(msg)
                    except Exception:
                        logger.exception(f"Could not process message {uid}")
        except Exception:
            logger.exception("Could not fetch unseen messages")
        return fetched_messages

    def fetch_part_ref(self, part: MessagePartRef) -> bytes | None:
        try:
            with self.imap_client as server:
                server.select_folder(part.folder)
                response = server.fetch([part.uid], [f"BODY.PEEK[{part.section}]"])
                data = response[part.uid][f"BODY[{part.section}]".encode("ascii")]
                return decode_transfer_encoding(data, part.encoding)
        except Exception:
            logger.exception(f"Failed to fetch part ref {part}")

    def list_folders(self) -> list[Folder]:
        try:
            with self.imap_client as server:
                return [Folder(
                    flags=[bytes2str(f) for f in folder_data[0]],
                    delimiter=bytes2str(folder_data[1]),
                    name=folder_data[2],
                ) for folder_data in server.list_folders()]
        except Exception:
            logger.exception(f"Failed to list folders")
        return []

    @property
    def imap_client(self):
        imap_client = IMAPClient(
            self.imap_host, ssl=True, ssl_context=ssl.create_default_context(cafile=certifi.where())
        )
        imap_client.use_uid = True
        imap_client.login(self.imap_username, self.imap_password)
        return imap_client

def bytes2str(b: bytes) -> str:
    return b.decode("utf-8", "replace")

def classify_parts(folder: str, uid: int, body_structure) -> (
        tuple[list[MessagePartRef], list[MessagePartRef], list[MessagePartRef], list[MessagePartRef]]
):
    html_parts: list[MessagePartRef] = []
    plain_parts: list[MessagePartRef] = []
    calendar_parts: list[MessagePartRef] = []
    attachment_parts: list[MessagePartRef] = []
    for section, part in iter_leaf_parts(body_structure):
        # As per https://datatracker.ietf.org/doc/html/rfc3501#section-7.4.2
        # leaf tuple: (type, subtype, params, id, desc, encoding, size, lines, md5, disposition, language, location)
        content_type = f"{bytes2str(part[0]).lower()}/{bytes2str(part[1]).lower()}"
        params = parse_params(part[2] if len(part) > 2 else None)
        disposition, disposition_params = parse_disposition(part[9] if len(part) > 9 else None)
        filename = disposition_params.get("filename") or params.get("name")
        ref = MessagePartRef(
            folder=folder,
            uid=uid,
            section=section,
            content_type=content_type,
            charset=params.get("charset").lower(),
            encoding=(bytes2str(part[5]).lower() if len(part) > 5 and part[5] else None),
            size=int(part[6]) if len(part) > 6 and isinstance(part[6], int) else None,
            filename=filename,
        )
        logger.debug(ref)
        if content_type == "text/html":
            html_parts.append(ref)
        elif content_type == "text/plain":
            plain_parts.append(ref)
        elif content_type == "text/calendar":
            calendar_parts.append(ref)
        # Attachments (explicit or inline-with-filename)
        if disposition == "attachment" or (disposition == "inline" and filename):
            attachment_parts.append(ref)
    return html_parts, plain_parts, calendar_parts, attachment_parts

def choose_largest_part(parts: list[MessagePartRef]) -> MessagePartRef:
    return max(parts, key=lambda m: m.size if m.size is not None else 0)

def decode_transfer_encoding(data: bytes, encoding: str | None) -> bytes:
    if not encoding:
        return data
    match encoding.lower():
        case "base64":
            return base64.b64decode(data)
        case "quoted-printable":
            return quopri.decodestring(data)
    # 7bit/8bit/binary typically require no decoding
    return data

def get_email_reader() -> EmailReader:
    imap_host = os.environ.get('IMAP_HOST')
    imap_username = os.environ.get('IMAP_USERNAME')
    imap_password = os.environ.get('IMAP_PASSWORD')
    if imap_host and imap_username and imap_password:
        return EmailReader(imap_host, imap_username, imap_password)
    raise RuntimeError("Unable to initialize EmailReader")

def iter_leaf_parts(body_structure, prefix: str = ""):
    if body_structure and isinstance(body_structure, tuple) and isinstance(body_structure[0], list):
        # multipart
        for i, child in enumerate(body_structure[0], start=1):
            section = f"{prefix}.{i}" if prefix else str(i)
            yield from iter_leaf_parts(child, section)
    else:
        # leaf
        yield prefix or "TEXT", body_structure

def parse_disposition(disposition) -> tuple[str | None, dict[str, str]]:
    # disposition is usually None or (b'ATTACHMENT', (b'FILENAME', b'x.pdf')) or (b'INLINE', (...))
    if not disposition:
        return None, {}
    if isinstance(disposition, tuple) and disposition:
        disposition_name = bytes2str(disposition[0])
        disposition_params = parse_params(disposition[1]) if len(disposition) > 1 else {}
        return (disposition_name.lower() if disposition_name else None), disposition_params
    return bytes2str(disposition), {}

def parse_header(blob: str) -> dict[str, str]:
    headers = HeaderParser().parsestr(blob)
    return {k.lower(): v for k, v in headers.items()}

def parse_params(params: tuple) -> dict[str, str]:
    # params looks like (b'CHARSET', b'UTF-8', b'NAME', b'file.txt') or None
    param_struct: dict[str, str] = {}
    if not params:
        return param_struct
    for i in range(0, len(params) - 1, 2):
        k = bytes2str(params[i]).lower()
        v = bytes2str(params[i + 1])
        param_struct[k] = v
    return param_struct

def parse_references(blob: str) -> list[str]:
    if not blob:
        return []
    references_parts = [r.strip() for r in blob.split()]
    return [r for r in references_parts if r]

if __name__ == "__main__":
    from dotenv import load_dotenv
    from prokaryotes.utils import setup_logging

    load_dotenv()
    setup_logging()

    reader = get_email_reader()
    logger.info(reader.list_folders())
    messages = reader.fetch_messages(criteria="UNSEEN")
    logger.info(messages)
    if messages:
        msg = messages.pop()
        body_data = reader.fetch_part_ref(msg.body)
        logger.info(body_data.decode(msg.body.charset).strip())
