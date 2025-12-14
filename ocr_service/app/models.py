import uuid
from sqlalchemy import Column, String, Text, DateTime, func, JSON
from sqlalchemy.dialects.postgresql import UUID

from .db import Base

class OCRTask(Base):
    __tablename__ = "ocr_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(20), nullable=False, default="pending")  # pending|processing|success|failed

    image_path = Column(Text, nullable=False)

    title = Column(Text, nullable=True)
    code = Column(Text, nullable=True)
    date = Column(String(10), nullable=True)  # ISO string YYYY-MM-DD

    raw_text = Column(Text, nullable=True)
    raw_lines = Column(JSON, nullable=True)

    error = Column(Text, nullable=True)
    meta = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
