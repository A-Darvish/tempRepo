from fastapi import FastAPI
from app.api import router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="OCR Field Extraction Service")
app.include_router(router)
