import threading
import time
import urllib.request
import json
from app_ui import run_server

t = threading.Thread(target=run_server, daemon=True)
t.start()

time.sleep(3)

# Test GET /api/samples
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/api/samples") as response:
        samples = json.loads(response.read().decode())
        print("API SAMPLES OK:", list(samples.keys()))
    
    # Test GET /api/predict_sample?key=sample_happy
    with urllib.request.urlopen("http://127.0.0.1:8000/api/predict_sample?key=sample_happy") as response:
        res = json.loads(response.read().decode())
        print("PREDICTION RESULT FOR HAPPY:")
        print("  Ground Truth Text :", res["ground_truth"])
        print("  Predicted Text    :", res["transcription"])
        print("  Predicted Emotion :", res["emotion"], f"({res['confidence']}%)")
        print("  CTC Loss          :", res["ctc_loss"])
        print("  CER (%)           :", res["metrics"]["cer"])
        print("  WER (%)           :", res["metrics"]["wer"])
        print("  UA (%)            :", res["metrics"]["ua"])
        print("  UWA (%)           :", res["metrics"]["uwa"])
        print("TEST PASSED SUCCESSFULLY!")
except Exception as e:
    print("Error:", e)
