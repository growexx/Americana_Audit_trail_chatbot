"""
Schema for upload request for implementing RAG
"""
from pydantic import BaseModel

class FilePushRequest(BaseModel):
    """
    File upload  Request only requires file_name and chat_id
    """
    chat_id: str
    file_name:str
