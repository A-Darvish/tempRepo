from sqlalchemy.orm import Session

from worker.celery_app import celery_app
from app.db import SessionLocal
from app.models import OCRTask
from app.services.ocr_pipeline import process_image_fast



@celery_app.task(bind=True, name="process_ocr_task", max_retries=2)
def process_ocr_task(self, task_id: str):
    db: Session = SessionLocal()
    try:
        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if not task:
            return {"status": "failed", "error": "Task not found"}

        task.status = "processing"
        db.commit()

        result = process_image_fast(task.image_path)

        if result.get("status") == "success":
            task.status = "success"
            task.title = result.get("title")
            task.code = result.get("code")
            task.date = result.get("date")
            task.raw_text = result.get("raw_text")
            task.raw_lines = result.get("raw_lines")
            task.meta = {
                "variant_used": result.get("variant_used"),
                "confidence": result.get("confidence"),
            }
            task.error = None
        else:
            task.status = "failed"
            task.error = result.get("error", "Unable to extract required fields.")
            task.raw_text = result.get("raw_text")
            task.raw_lines = result.get("raw_lines")
            task.meta = {"variant_used": result.get("variant_used")}

        db.commit()
        return {"status": task.status}

    except Exception as e:
        db.rollback()
        # mark failed
        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if task:
            task.status = "failed"
            task.error = f"Unhandled error: {repr(e)}"
            db.commit()
        raise
    finally:
        db.close()
