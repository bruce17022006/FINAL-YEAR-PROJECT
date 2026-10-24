@echo off
echo ======================================================================
echo Launching Multimodal Speech Recognition & Emotion Recognition Web UI
echo Conformer ASR + LLaMA-3.2-1B Sinkhorn Distillation + IEMOCAP SER
echo ======================================================================
echo.
echo --> Loading PyTorch Models into GPU...
echo --> Web Interface will be available at: http://127.0.0.1:8000
echo --> Open your browser and navigate to: http://127.0.0.1:8000
echo.
.\my_env\Scripts\python.exe app_ui.py
pause
