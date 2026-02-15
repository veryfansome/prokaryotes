from pydantic import BaseModel

class EmailMessage(BaseModel):
    id: int
    subject: str
    sender: str
    timestamp: str
    body: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
