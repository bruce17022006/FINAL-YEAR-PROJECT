@echo off
cd /d "%~dp0"
title Multimodal Speech & Emotion Recognition Web UI
echo ======================================================================
echo Launching Multimodal Speech Recognition and Emotion Recognition Web UI
echo Conformer ASR + LLaMA-3.2-1B Sinkhorn Distillation + IEMOCAP SER
echo ======================================================================
echo.
echo --> Working Folder: %CD%
echo --> Starting Web Server at http://localhost:8000 ...
echo.
if exist ".\my_env\Scripts\python.exe" (
    .\my_env\Scripts\python.exe -u app_ui.py
) else (
    python -u app_ui.py
)
pause
