@echo off
title Bus Passenger-Demand Forecasting — Launcher
cd /d "%~dp0"
set PYTHONPATH=%cd%

echo Starting FastAPI REST Service on http://localhost:8000 ...
start "Bus Demand FastAPI (Port 8000)" cmd /k ".\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000"

echo Starting Streamlit Decision-Support Dashboard on http://localhost:8501 ...
start "Bus Demand Streamlit (Port 8501)" cmd /k ".\.venv\Scripts\python.exe -m streamlit run app/dashboard.py --server.port 8501 --server.address 127.0.0.1"

echo.
echo Both services are launching!
echo  - Interactive Dashboard : http://localhost:8501
echo  - REST API Swagger Docs : http://localhost:8000/docs
echo.
pause
