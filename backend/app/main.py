from fastapi import FastAPI

app = FastAPI(
    title="AI Trading Lab API",
    description="Backend API for the AI Trading Lab project.",
    version="0.1.0",
)


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