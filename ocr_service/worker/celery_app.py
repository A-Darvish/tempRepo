import os
from dotenv import load_dotenv
load_dotenv()

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ocr_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# ✅ IMPORTANT: make Celery discover tasks
celery_app.autodiscover_tasks(["worker"])
