import logging
from fastapi import FastAPI

from app.routers import features, prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="AI Trading Lab API",
    description="Backend API for the AI Trading Lab project.",
    version="0.1.0",
)

app.include_router(prices.router)
app.include_router(features.router)

@app.get("/")
def read_root():
    return {
        "message": "AI Trading Lab API is running",
        "status": "ok",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }
  