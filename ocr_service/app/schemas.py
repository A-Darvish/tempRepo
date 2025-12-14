from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    id: str
    status: str

class OCRResultResponse(BaseModel):
    id: str
    status: str
    title: Optional[str] = None
    date: Optional[str] = None
    code: Optional[str] = None
    error: Optional[str] = None
