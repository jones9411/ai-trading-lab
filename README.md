@"

# AI Trading Lab

A full-stack learning project for building an AI-powered stock analysis, prediction, backtesting, and paper-trading platform.

## Goals

- Learn Python backend development
- Learn React/Next.js frontend development
- Learn machine learning and deep learning
- Build prediction models for stock market data
- Backtest trading strategies
- Simulate trades safely before considering any live trading

## Rule

This project is for education and paper trading only.
"@ | Set-Content README.md

## Start up server

cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload

Check api with http://127.0.0.1:8000/docs
