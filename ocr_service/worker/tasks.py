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

        task.status = "processing" # type: ignore
        db.commit()

        result = process_image_fast(task.image_path) # pyright: ignore[reportArgumentType]

        if result.get("status") == "success":
            task.status = "success" # type: ignore
            task.title = result.get("title") # type: ignore
            task.code = result.get("code") # type: ignore
            task.date = result.get("date") # type: ignore
            task.raw_text = result.get("raw_text") # type: ignore
            task.raw_lines = result.get("raw_lines") # type: ignore
            task.meta = { # type: ignore
                "variant_used": result.get("variant_used"),
                "confidence": result.get("confidence"),
            }
            task.error = None # type: ignore
        else:
            task.status = "failed" # type: ignore
            task.error = result.get("error", "Unable to extract required fields.")
            task.raw_text = result.get("raw_text") # type: ignore
            task.raw_lines = result.get("raw_lines") # type: ignore
            task.meta = {"variant_used": result.get("variant_used")} # type: ignore

        db.commit()
        return {"status": task.status}

    except Exception as e:
        db.rollback()
        # mark failed
        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if task:
            task.status = "failed" # type: ignore
            task.error = f"Unhandled error: {repr(e)}" # type: ignore
            db.commit()
        raise
    finally:
        db.close()
