import os
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from .db import get_db
from .models import OCRTask
from .schemas import UploadResponse, OCRResultResponse
from worker.tasks import process_ocr_task

router = APIRouter()

STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage/images")
os.makedirs(STORAGE_DIR, exist_ok=True)

@router.post("/api/upload-image/", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        ext = ".jpg"

    file_id = str(uuid.uuid4())
    save_path = os.path.join(STORAGE_DIR, f"{file_id}{ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    rec = OCRTask(status="pending", image_path=save_path)
    db.add(rec)
    db.commit()
    db.refresh(rec)

    # enqueue background job
    process_ocr_task.delay(str(rec.id))

    return UploadResponse(id=str(rec.id), status=rec.status)

@router.get("/api/ocr-tasks/{task_id}/result/", response_model=OCRResultResponse)
def get_result(task_id: str, db: Session = Depends(get_db)):
    rec = db.query(OCRTask).filter(OCRTask.id == task_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Task not found")

    resp = OCRResultResponse(
        id=str(rec.id),
        status=rec.status,
        title=rec.title,
        date=rec.date,
        code=rec.code,
        error=rec.error,
    )
    return resp
