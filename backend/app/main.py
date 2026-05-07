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
    
@app.get("/api/prices/{symbol}")
def get_prices(symbol: str):
    return{
        "symbol":symbol.upper(),
        "prices": [
            {
                "date": "2026-04-01",
                "open": 188.20,
                "high": 191.10,
                "low": 187.50,
                "close": 190.25,
                "volume": 53200000,
            },
            {
                "date": "2026-04-02",
                "open": 190.25,
                "high": 193.40,
                "low": 189.80,
                "close": 192.75,
                "volume": 48700000,
            },
            {
                "date": "2026-04-03",
                "open": 192.75,
                "high": 194.20,
                "low": 190.60,
                "close": 191.10,
                "volume": 50100000,
            },
        ],
    }
    
    
    