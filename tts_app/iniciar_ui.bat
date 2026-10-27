@echo off
cd /d "%~dp0.."
tts_app\.venv_ui\Scripts\python.exe -m streamlit run tts_app/app.py
