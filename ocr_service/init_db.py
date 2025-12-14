from app.db import engine, Base
import app.models  # ensures OCRTask is registered

Base.metadata.create_all(bind=engine)
print("✅ Tables created.")
