import os
import sys
import glob
import torch
import torchaudio
import sentencepiece as spm
from ctc_beam_search import CTCBeamSearchDecoder
from train_stage1_asr_generalized import StudentAcousticModel, TOKENIZER_MODEL, ctc_decode

sys.stdout.reconfigure(encoding='utf-8')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints_stage2/best_student_asr_stage2.pt"

def load_english_lexicon():
    words = set()
    # Extract from all transcripts
    for tf in glob.glob("Data/dev-clean/LibriSpeech/dev-clean/**/*.trans.txt", recursive=True):
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    for w in parts[1].lower().split():
                        words.add(w)

    # Also load from train transcripts if present
    if os.path.isfile("train_transcripts.txt"):
        with open("train_transcripts.txt", "r", encoding="utf-8") as f:
            for line in f:
                for w in line.strip().lower().split():
                    words.add(w)

    print(f"--> Loaded {len(words)} English lexicon dictionary words.")
    return words

def main():
    print("=" * 80)
    print("TESTING CTC PREFIX BEAM SEARCH WITH LEXICON DICTIONARY SCORING")
    print("=" * 80)

    lexicon = load_english_lexicon()
    decoder = CTCBeamSearchDecoder(
        sp_model_path=TOKENIZER_MODEL,
        lexicon_words=lexicon,
        beam_width=16,
        blank_penalty=1.2,
        word_bonus=1.5
    )

    model = StudentAcousticModel(
        input_dim=80, encoder_dim=256, num_encoder_layers=4,
        num_heads=4, vocab_size=1024, teacher_dim=2048, dropout=0.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
    )

    test_audios = [
        ("test_sample.wav", "she could get along without me and i could get along without her in the world"),
        ("test_sample_2.wav", "there are plenty of people who can attend closely to their business")
    ]

    for audio_path, ref in test_audios:
        if not os.path.isfile(audio_path):
            continue

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = mel_transform(waveform).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_mel = log_mel.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, _ = model(log_mel)
            # 1. Standard Greedy
            greedy_ids = logits.argmax(dim=-1)[0].cpu().tolist()
            greedy_pred = ctc_decode(greedy_ids, decoder.blank_id, decoder.sp)

            # 2. CTC Beam Search with Lexicon
            beam_pred = decoder.decode(logits[0])

        print(f"\n[Audio]: {audio_path}")
        print(f"  GROUND TRUTH : {ref.upper()}")
        print(f"  GREEDY DECODE: {greedy_pred.upper()}")
        print(f"  BEAM SEARCH  : {beam_pred.upper()}")

if __name__ == "__main__":
    main()
