import urllib.request
import json

for key in ['sample_happy', 'sample_sad', 'sample_angry', 'sample_neutral', 'sample_xxx']:
    try:
        url = f'http://127.0.0.1:8000/api/predict_sample?key={key}'
        raw = urllib.request.urlopen(url).read().decode()
        res = json.loads(raw)
        print(f"--- {key} ---")
        print("  CTC Loss:", res.get("ctc_loss"))
        print("  Metrics:", res.get("metrics"))
        print("  Transcription:", res.get("transcription"))
        print("  Ground Truth:", res.get("ground_truth"))
        print("  Emotion:", res.get("emotion"), res.get("confidence"))
    except Exception as e:
        print(f"ERROR on {key}:", e)
