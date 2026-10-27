@echo off
cd /d "%~dp0.."
python -m uvicorn api:app --app-dir tts_app --port 8000
