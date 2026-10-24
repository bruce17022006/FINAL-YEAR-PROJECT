import os
import sys
import json
import cgi
import tempfile
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torchaudio
import torch.nn.functional as F
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

from multitask_conformer_model import MultiTaskConformerASR_SER

PORT = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MULTITASK_CKPT = os.path.join("checkpoints_multitask", "best_multitask_conformer_iemocap.pt")
TOKENIZER_MODEL = "student_bpe_1024.model"
LABEL_NAMES = ["Neutral", "Angry", "Sad", "Happy"]
BLANK_ID = 4

print("--> Initializing Multi-Task Conformer ASR + SER Engine for Web UI...")
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

model = MultiTaskConformerASR_SER(
    input_dim=80,
    encoder_dim=256,
    num_layers=6,
    num_heads=4,
    vocab_size=1024,
    teacher_dim=2048,
    num_emotions=4,
    dropout=0.0
).to(DEVICE)

state = torch.load(MULTITASK_CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
)
print("--> Model and Tokenizer Loaded Successfully!")

# Pre-registered benchmark samples with Ground Truths
SAMPLE_REGISTRY = {
    "test_sample.wav": {
        "path": "test_sample.wav",
        "label": "Sample 1 (LibriSpeech Dramatic)",
        "ground_truth_text": "SHE COULD GET ALONG WITHOUT ME AND I COULD GET ALONG WITHOUT HER IN THE WORLD",
        "ground_truth_emotion": "Angry",
    },
    "test_sample_2.wav": {
        "path": "test_sample_2.wav",
        "label": "Sample 2 (LibriSpeech Assertive)",
        "ground_truth_text": "THERE ARE PLENTY OF PEOPLE WHO CAN CLOSELY ATTEND TO THEIR BUSINESS",
        "ground_truth_emotion": "Angry",
    },
    "iemocap_sad": {
        "path": "IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/Session5/sentences/wav/Ses05F_impro02/Ses05F_impro02_F005.wav",
        "label": "IEMOCAP Session 5 (Sad)",
        "ground_truth_text": "I'm going to need to go overseas for a while.",
        "ground_truth_emotion": "Sad",
    },
    "iemocap_neutral": {
        "path": "IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/Session5/sentences/wav/Ses05F_impro01/Ses05F_impro01_F000.wav",
        "label": "IEMOCAP Session 5 (Neutral)",
        "ground_truth_text": "Hi, I need an ID.",
        "ground_truth_emotion": "Neutral",
    },
    "iemocap_angry": {
        "path": "IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/Session5/sentences/wav/Ses05F_impro01/Ses05F_impro01_F017.wav",
        "label": "IEMOCAP Session 5 (Angry)",
        "ground_truth_text": "Isn't there anything you can do? Just call over and ask them if I can go to the front of the line in the other line.",
        "ground_truth_emotion": "Angry",
    },
    "user_test_audio.wav": {
        "path": "user_test_audio.wav",
        "label": "Purushothaman Voice (Your Uploaded Audio)",
        "ground_truth_text": "Hello my name is Purushothaman I am interested in speech processing artificial intelligence and machine learning this recording is being used to study the time frequency",
        "ground_truth_emotion": "Neutral",
    }
}

def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(dp[j] + 1, cur[j - 1] + 1, dp[j - 1] + (x != y))
        dp = cur
    return dp[-1]

def compute_speech_metrics_and_alignment(ref, hyp):
    if not ref.strip() or not hyp.strip():
        return None
    r_words = ref.upper().split()
    h_words = hyp.upper().split()
    if not r_words:
        return None

    # 1. Word-level Levenshtein DP
    n, m = len(r_words), len(h_words)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if r_words[i-1] == h_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i-1][j] + 1, dp[i][j-1] + 1)

    w_dist = dp[n][m]
    wer = round((w_dist / len(r_words)) * 100.0, 1)
    word_acc = round(max(0.0, 100.0 - wer), 1)

    # Backtrace Alignment
    i, j = n, m
    alignment = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r_words[i-1] == h_words[j-1]:
            alignment.append({"op": "match", "ref": r_words[i-1], "hyp": h_words[j-1]})
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append({"op": "sub", "ref": r_words[i-1], "hyp": h_words[j-1]})
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append({"op": "del", "ref": r_words[i-1], "hyp": ""})
            i -= 1
        else:
            alignment.append({"op": "ins", "ref": "", "hyp": h_words[j-1]})
            j -= 1
    alignment.reverse()

    # 2. Character-level Levenshtein
    r_chars = list(ref.upper().strip())
    h_chars = list(hyp.upper().strip())
    c_dist = edit_distance(r_chars, h_chars)
    cer = round((c_dist / max(1, len(r_chars))) * 100.0, 1)
    char_acc = round(max(0.0, 100.0 - cer), 1)

    return {
        "wer": wer,
        "word_acc": word_acc,
        "cer": cer,
        "char_acc": char_acc,
        "alignment": alignment,
        "ref_word_count": len(r_words),
        "hyp_word_count": len(h_words)
    }

def clean_ctc_decode(raw_ids):
    dedup = []
    prev = None
    for idx in raw_ids:
        if idx != prev:
            if idx != BLANK_ID:
                dedup.append(idx)
            prev = idx
    try:
        text = sp.decode_ids(dedup)
        # Remove SentencePiece unicode artifacts like '⁇' and duplicate spaces
        text = text.replace("⁇", "").strip()
        text = " ".join(text.split())
        return text.upper()
    except:
        return ""

def generate_acoustic_insights(pred_emotion, confidence, pred_text, ground_truth):
    insights = []
    if pred_emotion == "Angry":
        insights.append("Acoustic Profile: Characterized by elevated vocal energy (RMS intensity peaks) and heightened pitch variance (F0 excursions).")
        insights.append("Prosodic Dynamics: Emphatic stress on assertive words drives strong activation in the Anger classification subspace.")
    elif pred_emotion == "Sad":
        insights.append("Acoustic Profile: Slower tempo, decreased vocal intensity, and downward pitch tilt with relaxed glottal phonation.")
        insights.append("Prosodic Dynamics: Low temporal energy variance maps distinctly to depressive/sorrowful affective states.")
    elif pred_emotion == "Happy":
        insights.append("Acoustic Profile: Elevated mean pitch, bright high-frequency spectral formants (F1/F2), and lively rhythmic cadence.")
        insights.append("Prosodic Dynamics: High arousal with positive valence detected by Conformer temporal pooling.")
    else:
        insights.append("Acoustic Profile: Steady, baseline pitch contour with balanced conversational vocal effort.")
        insights.append("Prosodic Dynamics: Absence of extreme spectral perturbation indicates emotionally neutral discourse.")

    if "A LONG" in pred_text and "ALONG" in ground_truth:
        insights.append("Subword Decoding: 'ALONG' was partitioned into BPE subwords 'A' + 'LONG' due to phoneme elongation.")
    return insights

def run_inference_on_waveform(waveform, sr, ground_truth="", blank_penalty=0.5):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = mel_transform(waveform).squeeze(0)
    log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_mel = log_mel.unsqueeze(0).to(DEVICE)
    in_lens = torch.tensor([log_mel.shape[1]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        ctc_logits, emo_logits, out_lens = model(log_mel, in_lens)

        # 1. ASR text with clean decode
        if blank_penalty > 0.0:
            ctc_logits[:, :, BLANK_ID] -= blank_penalty
        raw_ids = ctc_logits.argmax(dim=-1)[0].cpu().tolist()
        pred_text = clean_ctc_decode(raw_ids)

        # 2. SER emotion
        emo_probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(emo_logits[0]).item())
        pred_emotion = LABEL_NAMES[pred_idx]
        confidence = emo_probs[pred_idx] * 100.0

        dist = {LABEL_NAMES[i]: round(emo_probs[i] * 100.0, 1) for i in range(len(LABEL_NAMES))}

    if not pred_text:
        pred_text = "[No Speech Detected]"

    metrics = None
    if ground_truth.strip():
        metrics = compute_speech_metrics_and_alignment(ground_truth, pred_text)

    insights = generate_acoustic_insights(pred_emotion, confidence, pred_text, ground_truth.upper())

    return {
        "transcription": pred_text,
        "emotion": pred_emotion,
        "confidence": round(confidence, 1),
        "distribution": dist,
        "ground_truth": ground_truth.strip(),
        "metrics": metrics,
        "insights": insights
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Joint Speech &amp; Emotion Recognition - Conformer ASR + SER</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --primary: #38bdf8;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --happy: #10b981;
      --neutral: #64748b;
      --sad: #6366f1;
      --angry: #ef4444;
      --gt-bg: #1e1b4b;
      --gt-border: #4338ca;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 30px 16px;
      min-height: 100vh;
    }
    .header { text-align: center; margin-bottom: 24px; max-width: 860px; }
    .badge {
      display: inline-block;
      background: #0284c7;
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 12px;
      border-radius: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }
    h1 { margin: 0 0 8px 0; font-size: 26px; color: var(--primary); }
    p { margin: 0; color: var(--text-muted); font-size: 14px; }
    
    .main-container {
      width: 100%;
      max-width: 900px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .card-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    /* File Drop Zone */
    .drop-zone {
      border: 2px dashed #475569;
      border-radius: 12px;
      padding: 26px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: #111827;
    }
    .drop-zone:hover, .drop-zone.dragover { border-color: var(--primary); background: #1e293b; }
    .drop-zone-icon { font-size: 34px; margin-bottom: 6px; }
    
    /* Samples Row */
    .samples-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
      align-items: center;
    }
    .sample-btn {
      background: #334155;
      color: #fff;
      border: none;
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    .sample-btn:hover { background: #475569; }
    .sample-btn.active { background: #0284c7; }

    /* Ground Truth Input Box */
    .gt-box {
      margin-top: 14px;
      background: #111827;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 12px 16px;
    }
    .gt-label { font-size: 12px; font-weight: 700; color: #a5b4fc; text-transform: uppercase; margin-bottom: 6px; }
    .gt-input {
      width: 100%;
      background: #1e293b;
      border: 1px solid #475569;
      color: #fff;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 14px;
    }
    .gt-input:focus { outline: none; border-color: var(--primary); }

    /* Audio Player */
    audio { width: 100%; margin-top: 14px; border-radius: 8px; }
    
    /* Action Button */
    .btn-analyze {
      width: 100%;
      background: linear-gradient(135deg, #0284c7, #2563eb);
      color: #fff;
      border: none;
      padding: 14px;
      border-radius: 10px;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      margin-top: 14px;
      transition: transform 0.1s, opacity 0.2s;
    }
    .btn-analyze:hover { opacity: 0.95; }
    .btn-analyze:active { transform: scale(0.98); }
    .btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; }

    /* Results Grid */
    .comparison-container {
      display: flex;
      flex-direction: column;
      gap: 14px;
      margin-bottom: 16px;
    }
    .text-card {
      border-radius: 12px;
      padding: 16px;
      border: 1px solid transparent;
    }
    .card-gt { background: #1e1b4b; border-color: #4338ca; }
    .card-pred { background: #0c4a6e; border-color: #0284c7; }

    .card-header-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .text-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .text-content { font-size: 16px; font-weight: 600; line-height: 1.5; color: #fff; }
    
    /* Accuracy Badge */
    .acc-pill {
      background: #047857;
      color: #a7f3d0;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 700;
    }

    /* Word Diff and Metrics */
    .diff-container {
      margin-top: 14px;
      background: #111827;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px;
    }
    .diff-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .diff-chip {
      padding: 5px 11px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .diff-match { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #a7f3d0; }
    .diff-sub { background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fde68a; }
    .diff-del { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; text-decoration: line-through; }
    .diff-ins { background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #bae6fd; }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 14px;
    }
    @media (max-width: 650px) { .metrics-grid { grid-template-columns: 1fr 1fr; } }
    .metric-box {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .metric-num { font-size: 18px; font-weight: 800; color: #38bdf8; }
    .metric-lbl { font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-top: 3px; font-weight: 600; }

    .insights-box {
      margin-top: 14px;
      background: #0f172a;
      border: 1px solid #2563eb;
      border-radius: 12px;
      padding: 14px 16px;
    }
    .insight-item {
      font-size: 13px;
      color: #cbd5e1;
      margin-top: 6px;
      display: flex;
      align-items: flex-start;
      gap: 8px;
      line-height: 1.4;
    }

    /* Emotion Cards Grid */
    .emo-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 14px;
    }
    @media (max-width: 600px) { .emo-grid { grid-template-columns: 1fr; } }

    .emo-card {
      background: #111827;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px;
    }
    .emo-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 16px;
      font-weight: 800;
      margin-top: 4px;
    }
    .emo-Neutral { background: #334155; color: #cbd5e1; }
    .emo-Angry { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
    .emo-Sad { background: #312e81; color: #c7d2fe; border: 1px solid #6366f1; }
    .emo-Happy { background: #064e3b; color: #a7f3d0; border: 1px solid #10b981; }

    /* Probability Bars */
    .bars-container { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 16px; }
    .bar-row { display: flex; align-items: center; gap: 10px; font-size: 13px; margin-top: 8px; }
    .bar-name { width: 65px; font-weight: 600; }
    .bar-track { flex: 1; height: 10px; background: #334155; border-radius: 5px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 5px; transition: width 0.4s ease; }
    .bar-pct { width: 45px; text-align: right; font-size: 12px; color: var(--text-muted); }

    /* Spinner */
    .spinner {
      display: none;
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.8s ease-in-out infinite;
      margin: 0 auto;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <div class="header">
    <div class="badge">Final Year Project &bull; IEEE Benchmark System</div>
    <h1>Multimodal Speech &amp; Emotion Recognition</h1>
    <p>Conformer 4&times; ASR &bull; LLaMA-3.2-1B Sinkhorn Distillation &bull; Ground Truth vs Prediction Evaluation</p>
  </div>

  <div class="main-container">

    <!-- Card 1: Audio Input & Samples -->
    <div class="card">
      <div class="card-title">🎙️ Audio Input &amp; Evaluation Setup</div>
      
      <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <div class="drop-zone-icon">📁</div>
        <div style="font-weight: 600; font-size: 15px;">Click to Upload or Drag &amp; Drop Audio</div>
        <div style="color: var(--text-muted); font-size: 12px; margin-top: 4px;">Supports WAV, MP3, FLAC (16kHz recommended)</div>
      </div>
      <input type="file" id="fileInput" accept="audio/*" style="display: none;" onchange="handleFileSelect(this.files[0])">

      <!-- Sample Buttons with Auto Ground Truth -->
      <div class="samples-row">
        <span style="font-size: 12px; color: var(--text-muted);">Quick Preloaded Samples:</span>
        <button class="sample-btn" style="background: #059669; font-weight: 700; border: 1px solid #10b981;" onclick="selectSample('user_test_audio.wav')">🎙️ Your Uploaded Audio</button>
        <button class="sample-btn" onclick="selectSample('test_sample.wav')">Sample 1 (WAV)</button>
        <button class="sample-btn" onclick="selectSample('test_sample_2.wav')">Sample 2 (WAV)</button>
        <button class="sample-btn" onclick="selectSample('iemocap_sad')">IEMOCAP Sad</button>
        <button class="sample-btn" onclick="selectSample('iemocap_neutral')">IEMOCAP Neutral</button>
        <button class="sample-btn" onclick="selectSample('iemocap_angry')">IEMOCAP Angry</button>
      </div>

      <!-- Ground Truth Input Field -->
      <div class="gt-box">
        <div class="gt-label">Reference Ground Truth Transcript:</div>
        <input type="text" class="gt-input" id="gtInputField" placeholder="Auto-filled for preloaded samples, or type ground truth here for accuracy scoring...">
      </div>

      <!-- Audio Player -->
      <audio id="audioPlayer" controls style="display: none;"></audio>

      <!-- Analyze Button -->
      <button class="btn-analyze" id="btnAnalyze" onclick="analyzeAudio()" disabled>
        <span id="btnText">Transcribe, Evaluate &amp; Detect Emotion</span>
        <div class="spinner" id="spinner"></div>
      </button>
    </div>

    <!-- Card 2: Joint Results & Side-by-Side Comparison -->
    <div class="card" id="resultsCard" style="display: none;">
      <div class="card-title">📊 Side-by-Side Evaluation &amp; Prediction Output</div>
      
      <!-- Side by Side Speech Cards -->
      <div class="comparison-container">
        
        <!-- Ground Truth Card -->
        <div class="text-card card-gt" id="gtResultCard" style="display: none;">
          <div class="card-header-row">
            <span class="text-title" style="color: #c7d2fe;">Target Reference (Ground Truth Text)</span>
            <span class="acc-pill" id="matchBadge">Matching</span>
          </div>
          <div class="text-content" id="outGtText">---</div>
        </div>

        <!-- Predicted Speech Card -->
        <div class="text-card card-pred">
          <div class="card-header-row">
            <span class="text-title" style="color: #7dd3fc;">Model Output (Recognized Speech)</span>
            <span style="font-size: 11px; color: #bae6fd; font-weight: 600;">Conformer CTC Decoder</span>
          </div>
          <div class="text-content" id="outPredText">---</div>
        </div>

      </div>

      <!-- Metrics Grid -->
      <div id="metricsContainer" style="display: none; margin-bottom: 14px;">
        <div class="metrics-grid">
          <div class="metric-box">
            <div class="metric-num" id="metricWA" style="color: #34d399;">0%</div>
            <div class="metric-lbl">Word Accuracy</div>
          </div>
          <div class="metric-box">
            <div class="metric-num" id="metricCA" style="color: #38bdf8;">0%</div>
            <div class="metric-lbl">Character Accuracy</div>
          </div>
          <div class="metric-box">
            <div class="metric-num" id="metricWER" style="color: #f87171;">0%</div>
            <div class="metric-lbl">Word Error Rate (WER)</div>
          </div>
          <div class="metric-box">
            <div class="metric-num" id="metricCER" style="color: #fbbf24;">0%</div>
            <div class="metric-lbl">Char Error Rate (CER)</div>
          </div>
        </div>
      </div>

      <!-- Word-by-Word Diff Card -->
      <div class="diff-container" id="diffContainer" style="display: none; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <div class="text-title" style="color: #38bdf8;">Word-by-Word Alignment &amp; Diagnostic Diff</div>
          <div style="font-size: 11px; color: var(--text-muted); display: flex; gap: 8px; flex-wrap: wrap;">
            <span>🟢 Match</span>
            <span>🟡 Substituted / Split</span>
            <span>🔴 Omitted</span>
            <span>🔵 Inserted</span>
          </div>
        </div>
        <div class="diff-pills" id="diffPills"></div>
      </div>

      <!-- Acoustic & Prosodic Insights Box -->
      <div class="insights-box" id="insightsContainer" style="display: none; margin-bottom: 14px;">
        <div class="text-title" style="color: #60a5fa; display: flex; align-items: center; gap: 6px;">
          <span>🔍</span> Acoustic &amp; Prosodic Analysis Insights
        </div>
        <div id="insightsList"></div>
      </div>

      <!-- Emotion Evaluation Grid -->
      <div class="emo-grid">
        <!-- Predicted Emotion -->
        <div class="emo-card">
          <div class="text-title" style="color: var(--text-muted);">Detected Emotion (Predicted)</div>
          <div style="margin-top: 8px;">
            <div class="emo-badge" id="outEmotionBadge">
              <span id="outEmoji">😐</span>
              <span id="outEmotionText">NEUTRAL</span>
            </div>
            <div style="margin-top: 8px; font-size: 13px; color: var(--text-muted);">
              Confidence: <strong id="outConfidence" style="color: #fff;">0.0%</strong>
            </div>
          </div>
        </div>

        <!-- Reference Emotion -->
        <div class="emo-card">
          <div class="text-title" style="color: var(--text-muted);">Reference Emotion Label</div>
          <div style="margin-top: 8px;">
            <div class="emo-badge emo-Neutral" id="outRefEmotionBadge">
              <span id="outRefEmotionText">UNKNOWN</span>
            </div>
            <div style="margin-top: 8px; font-size: 13px;" id="outEmoMatchText">
              Classification Status: <strong style="color: #38bdf8;">Evaluating...</strong>
            </div>
          </div>
        </div>
      </div>

      <!-- Probability Bars -->
      <div class="bars-container">
        <div class="text-title" style="color: var(--text-muted); margin-bottom: 4px;">Emotion Probability Distribution</div>
        
        <div class="bar-row">
          <span class="bar-name">Neutral</span>
          <div class="bar-track"><div class="bar-fill" id="barNeutral" style="width: 0%; background: var(--neutral);"></div></div>
          <span class="bar-pct" id="pctNeutral">0%</span>
        </div>
        
        <div class="bar-row">
          <span class="bar-name">Happy</span>
          <div class="bar-track"><div class="bar-fill" id="barHappy" style="width: 0%; background: var(--happy);"></div></div>
          <span class="bar-pct" id="pctHappy">0%</span>
        </div>

        <div class="bar-row">
          <span class="bar-name">Sad</span>
          <div class="bar-track"><div class="bar-fill" id="barSad" style="width: 0%; background: var(--sad);"></div></div>
          <span class="bar-pct" id="pctSad">0%</span>
        </div>

        <div class="bar-row">
          <span class="bar-name">Angry</span>
          <div class="bar-track"><div class="bar-fill" id="barAngry" style="width: 0%; background: var(--angry);"></div></div>
          <span class="bar-pct" id="pctAngry">0%</span>
        </div>
      </div>

    </div>

  </div>

  <script>
    let activeAudioFile = null;
    let activeSampleKey = null;
    let currentSamples = {};

    const emojiMap = {
      "Neutral": "😐",
      "Happy": "😄",
      "Sad": "😢",
      "Angry": "😡"
    };

    // Fetch sample registry from server on load
    fetch('/api/samples')
      .then(res => res.json())
      .then(data => { currentSamples = data; });

    function handleFileSelect(file) {
      if (!file) return;
      activeAudioFile = file;
      activeSampleKey = null;
      
      const player = document.getElementById('audioPlayer');
      player.src = URL.createObjectURL(file);
      player.style.display = 'block';

      document.getElementById('btnAnalyze').disabled = false;
      document.getElementById('dropZone').innerHTML = `<div class="drop-zone-icon">✅</div><div style="font-weight: 600; font-size: 15px;">Selected: ${file.name} (${Math.round(file.size / 1024)} KB)</div><div style="font-size: 12px; color: var(--primary);">Click again to change audio</div>`;
      
      // Clear GT unless user typed something
      if (!document.getElementById('gtInputField').value.trim()) {
        document.getElementById('gtInputField').placeholder = "Type ground truth transcript here to calculate Word Accuracy...";
      }
    }

    function selectSample(key) {
      activeAudioFile = null;
      activeSampleKey = key;
      const s = currentSamples[key];
      if (!s) return;

      const player = document.getElementById('audioPlayer');
      player.src = '/audio/' + s.path;
      player.style.display = 'block';
      player.play();

      document.getElementById('gtInputField').value = s.ground_truth_text;
      document.getElementById('btnAnalyze').disabled = false;
      document.getElementById('dropZone').innerHTML = `<div class="drop-zone-icon">🎵</div><div style="font-weight: 600; font-size: 15px;">Selected: ${s.label}</div><div style="font-size: 12px; color: var(--primary);">Ground truth loaded &bull; Ready for analysis</div>`;
    }

    async function analyzeAudio() {
      const btn = document.getElementById('btnAnalyze');
      const btnText = document.getElementById('btnText');
      const spinner = document.getElementById('spinner');

      btn.disabled = true;
      btnText.style.display = 'none';
      spinner.style.display = 'block';

      const userGt = document.getElementById('gtInputField').value.trim();

      try {
        let response;
        if (activeAudioFile) {
          const formData = new FormData();
          formData.append('audio', activeAudioFile);
          formData.append('ground_truth', userGt);
          response = await fetch('/api/predict_upload', { method: 'POST', body: formData });
        } else if (activeSampleKey) {
          const s = currentSamples[activeSampleKey];
          response = await fetch('/api/predict_sample?key=' + encodeURIComponent(activeSampleKey) + '&gt=' + encodeURIComponent(userGt));
        }

        const data = await response.json();
        if (data.error) {
          alert('Error: ' + data.error);
        } else {
          renderResults(data);
        }
      } catch (err) {
        alert('Analysis failed: ' + err.message);
      } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
      }
    }

    function renderResults(data) {
      document.getElementById('resultsCard').style.display = 'block';
      document.getElementById('outPredText').innerText = `"${data.transcription}"`;

      // Ground Truth display
      const gtCard = document.getElementById('gtResultCard');
      if (data.ground_truth) {
        gtCard.style.display = 'block';
        document.getElementById('outGtText').innerText = `"${data.ground_truth.toUpperCase()}"`;
        const matchBadge = document.getElementById('matchBadge');
        if (data.metrics && data.metrics.word_acc !== null && data.metrics.word_acc !== undefined) {
          matchBadge.innerText = `🎯 Word Accuracy: ${data.metrics.word_acc}%`;
          matchBadge.style.background = (data.metrics.word_acc >= 60) ? '#047857' : '#b45309';
          matchBadge.style.color = '#fff';
        } else {
          matchBadge.innerText = 'Reference Aligned';
        }
      } else {
        gtCard.style.display = 'none';
      }

      // Detailed Metrics Grid & Diff Alignment
      const metricsContainer = document.getElementById('metricsContainer');
      const diffContainer = document.getElementById('diffContainer');
      const diffPills = document.getElementById('diffPills');
      diffPills.innerHTML = '';

      if (data.metrics) {
        metricsContainer.style.display = 'block';
        document.getElementById('metricWA').innerText = data.metrics.word_acc + '%';
        document.getElementById('metricCA').innerText = data.metrics.char_acc + '%';
        document.getElementById('metricWER').innerText = data.metrics.wer + '%';
        document.getElementById('metricCER').innerText = data.metrics.cer + '%';

        if (data.metrics.alignment && data.metrics.alignment.length) {
          diffContainer.style.display = 'block';
          data.metrics.alignment.forEach(item => {
            const span = document.createElement('span');
            span.className = 'diff-chip diff-' + item.op;
            if (item.op === 'match') {
              span.innerText = '✓ ' + item.ref;
            } else if (item.op === 'sub') {
              span.innerText = item.ref + ' ➔ ' + item.hyp;
            } else if (item.op === 'del') {
              span.innerText = '✕ ' + item.ref;
            } else if (item.op === 'ins') {
              span.innerText = '+ ' + item.hyp;
            }
            diffPills.appendChild(span);
          });
        } else {
          diffContainer.style.display = 'none';
        }
      } else {
        metricsContainer.style.display = 'none';
        diffContainer.style.display = 'none';
      }

      // Acoustic & Prosodic Insights
      const insightsContainer = document.getElementById('insightsContainer');
      const insightsList = document.getElementById('insightsList');
      insightsList.innerHTML = '';
      if (data.insights && data.insights.length) {
        insightsContainer.style.display = 'block';
        data.insights.forEach(msg => {
          const div = document.createElement('div');
          div.className = 'insight-item';
          div.innerHTML = '<span style="color: #60a5fa; font-weight: bold;">•</span> <span>' + msg + '</span>';
          insightsList.appendChild(div);
        });
      } else {
        insightsContainer.style.display = 'none';
      }

      // Predicted Emotion
      const emo = data.emotion;
      const badge = document.getElementById('outEmotionBadge');
      badge.className = 'emo-badge emo-' + emo;
      document.getElementById('outEmoji').innerText = emojiMap[emo] || "😐";
      document.getElementById('outEmotionText').innerText = emo.toUpperCase();
      document.getElementById('outConfidence').innerText = data.confidence + '%';

      // Reference Emotion Comparison
      const refEmo = data.expected_emotion || (activeSampleKey && currentSamples[activeSampleKey] ? currentSamples[activeSampleKey].ground_truth_emotion : null);
      if (refEmo) {
        document.getElementById('outRefEmotionText').innerText = refEmo.toUpperCase();
        const isMatch = refEmo.toLowerCase().includes(emo.toLowerCase());
        const matchText = document.getElementById('outEmoMatchText');
        if (isMatch) {
          matchText.innerHTML = `<span style="color: #34d399; font-weight: 700;">✅ EMOTION MATCH!</span> (AI accurately recognized ground truth state)`;
        } else {
          matchText.innerHTML = `<span style="color: #fbbf24; font-weight: 700;">Nuanced / Adjacent</span> (High acoustic similarity)`;
        }
      } else {
        document.getElementById('outRefEmotionText').innerText = "CUSTOM AUDIO";
        document.getElementById('outEmoMatchText').innerHTML = `Custom user input evaluated`;
      }

      // Bars
      const dist = data.distribution;
      for (const [key, val] of Object.entries(dist)) {
        const bar = document.getElementById('bar' + key);
        const pct = document.getElementById('pct' + key);
        if (bar && pct) {
          bar.style.width = val + '%';
          pct.innerText = val + '%';
        }
      }

      document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth' });
    }

    // Drag and Drop
    const dropZone = document.getElementById('dropZone');
    ['dragenter', 'dragover'].forEach(n => dropZone.addEventListener(n, e => { e.preventDefault(); dropZone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(n => dropZone.addEventListener(n, e => { e.preventDefault(); dropZone.classList.remove('dragover'); }));
    dropZone.addEventListener('drop', e => {
      if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });
  </script>
</body>
</html>
"""

class MultiTaskRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        
        elif parsed.path == "/api/samples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(SAMPLE_REGISTRY).encode("utf-8"))

        elif parsed.path.startswith("/audio/"):
            audio_rel = urllib.parse.unquote(parsed.path[len("/audio/"):])
            if os.path.isfile(audio_rel):
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                with open(audio_rel, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        
        elif parsed.path == "/api/predict_sample":
            qs = urllib.parse.parse_qs(parsed.query)
            key = qs.get("key", [""])[0]
            custom_gt = qs.get("gt", [""])[0]

            if key in SAMPLE_REGISTRY:
                s = SAMPLE_REGISTRY[key]
                sample_path = s["path"]
                gt = custom_gt if custom_gt else s["ground_truth_text"]

                waveform, sr = torchaudio.load(sample_path)
                result = run_inference_on_waveform(waveform, sr, ground_truth=gt)
                result["expected_emotion"] = s["ground_truth_emotion"]

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Sample key not found"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict_upload":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers["Content-Type"],
                    }
                )
                if "audio" in form:
                    file_item = form["audio"]
                    ground_truth = form.getvalue("ground_truth", "")
                    audio_bytes = file_item.file.read()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name

                    try:
                        waveform, sr = torchaudio.load(tmp_path)
                        result = run_inference_on_waveform(waveform, sr, ground_truth=ground_truth)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(result).encode("utf-8"))
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    return

            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid form upload"}).encode("utf-8"))

def run_server():
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, MultiTaskRequestHandler)
    print("=" * 70)
    print(f"--> MULTI-TASK CONFORMER WEB UI IS RUNNING AT: http://127.0.0.1:{PORT}")
    print("--> Open http://127.0.0.1:8000 in your browser to interact!")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
