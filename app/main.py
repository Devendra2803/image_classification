from fastapi import FastAPI
from app.classification_feature.routes import router as classification_router

app = FastAPI(title="Image Classifier API")

# Register classifier routes
app.include_router(classification_router)

@app.get("/")
def root():
    return {"message": "Welcome to the Image Classifier API"}
