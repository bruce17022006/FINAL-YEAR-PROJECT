@echo off
cd /d "%~dp0"
title Multimodal Speech & Emotion Recognition Streamlit UI
echo ======================================================================
echo Launching Multimodal Speech Recognition and Emotion Recognition Streamlit UI
echo Conformer ASR + LLaMA-3.2-1B Sinkhorn Distillation + IEMOCAP SER
echo ======================================================================
echo.
echo --> Working Folder: %CD%
echo --> Starting Streamlit at http://localhost:8501 ...
echo.
if exist ".\my_env\Scripts\streamlit.exe" (
    .\my_env\Scripts\streamlit.exe run streamlit_app.py --server.port 8501 --server.headless true
) else (
    streamlit run streamlit_app.py --server.port 8501 --server.headless true
)
pause
