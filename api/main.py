from fastapi import FastAPI, HTTPException
import mlflow
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="ML Model API", description="API for serving ML model predictions")

@app.get("/")
def root():
    return {"message": "ML Model API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
