import os
import sys
import math
import torch
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForCausalLM

from conformer_asr import StudentConformerASR
from ctc_beam_search import CTCBeamSearchDecoder
from train_stage1_asr_generalized import TOKENIZER_MODEL, ctc_decode

sys.stdout.reconfigure(encoding='utf-8')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFORMER_DISTILLED_CKPT = os.path.join("checkpoints_conformer", "best_conformer_stage2_distilled.pt")
CONFORMER_BASE_CKPT = os.path.join("checkpoints_conformer", "best_student_conformer_4x.pt")
CONFORMER_CKPT = CONFORMER_DISTILLED_CKPT if os.path.isfile(CONFORMER_DISTILLED_CKPT) else CONFORMER_BASE_CKPT

class TwoPassRescorer:
    """
    Two-Pass Speech Recognition System:
    Pass 1: Conformer ASR (4x Subsampling) + CTC Beam Search produces Top-K candidates.
    Pass 2: Causal Language Model rescores candidates using perplexity / sequence likelihood.
    """
    def __init__(
        self,
        asr_checkpoint_path: str,
        sp_model_path: str,
        lm_model_id: str = "gpt2",
        beam_width: int = 8,
        lm_weight: float = 0.5,
    ):
        self.device = DEVICE
        self.lm_weight = lm_weight
        self.beam_width = beam_width

        # 1. Load Tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.blank_id = self.sp.piece_to_id("<blank>")

        # 2. Load 6-Layer Conformer Acoustic Model
        print("--> Loading Conformer Acoustic Model (4x Subsampling)...")
        self.asr_model = StudentConformerASR(
            input_dim=80,
            encoder_dim=256,
            num_layers=6,
            num_heads=4,
            vocab_size=1024,
            teacher_dim=2048,
            dropout=0.0
        ).to(self.device)

        state = torch.load(asr_checkpoint_path, map_location=self.device, weights_only=True)
        self.asr_model.load_state_dict(state, strict=True)
        self.asr_model.eval()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
        )

        # 3. Load Causal Language Model for Pass 2 Rescoring
        print(f"--> Loading Causal Language Model ({lm_model_id}) for Second-Pass Rescoring...")
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_id)
        self.lm_model = AutoModelForCausalLM.from_pretrained(lm_model_id).to(self.device)
        self.lm_model.eval()

        # 4. Beam Search Decoder for Pass 1
        self.decoder = CTCBeamSearchDecoder(
            sp_model_path=sp_model_path,
            beam_width=beam_width,
            blank_penalty=0.5,
            word_bonus=0.2
        )

    def compute_lm_score(self, text: str) -> float:
        """Computes log-likelihood of a candidate text under the Language Model."""
        if not text.strip():
            return -100.0
        inputs = self.lm_tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] <= 1:
            return 0.0

        with torch.no_grad():
            outputs = self.lm_model(input_ids, labels=input_ids)
            # loss is cross-entropy: -mean(log P). Total log P = -loss * sequence_length
            total_log_p = -outputs.loss.item() * (input_ids.shape[1] - 1)
        return total_log_p

    def get_top_k_candidates(self, logits: torch.Tensor, top_k: int = 4):
        """Extracts top candidate beam sequences and their acoustic scores."""
        T, V = logits.shape
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        beams = {(): (1.0, 0.0)}

        for t in range(T):
            step_probs = probs[t]
            top_indices = [
                i for i, p in enumerate(step_probs) if p > 0.005 or i == self.blank_id
            ]

            next_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                p_total = p_b + p_nb
                if p_total <= 0:
                    continue

                # Blank
                blank_prob = step_probs[self.blank_id]
                nb_b, nb_nb = next_beams.get(prefix, (0.0, 0.0))
                next_beams[prefix] = (nb_b + p_total * blank_prob, nb_nb)

                # Non-blank
                last = prefix[-1] if len(prefix) > 0 else None
                for c in top_indices:
                    if c == self.blank_id:
                        continue
                    prob_c = step_probs[c]
                    if c == last:
                        new_p = prefix + (c,)
                        b1, nb1 = next_beams.get(new_p, (0.0, 0.0))
                        next_beams[new_p] = (b1, nb1 + p_b * prob_c)

                        b2, nb2 = next_beams.get(prefix, (0.0, 0.0))
                        next_beams[prefix] = (b2, nb2 + p_nb * prob_c)
                    else:
                        new_p = prefix + (c,)
                        b1, nb1 = next_beams.get(new_p, (0.0, 0.0))
                        next_beams[new_p] = (b1, nb1 + p_total * prob_c)

            # Sort and keep top beam_width
            scored = []
            for prefix, (p_b, p_nb) in next_beams.items():
                tot = p_b + p_nb
                if tot > 0:
                    scored.append((math.log(tot + 1e-15), prefix, (p_b, p_nb)))
            scored.sort(key=lambda x: x[0], reverse=True)
            beams = {p: probs_tup for _, p, probs_tup in scored[: self.beam_width]}

        # Form unique candidate strings
        candidates = []
        seen_texts = set()
        for prefix in beams:
            text = self.sp.decode_ids(list(prefix)).strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                acoustic_log_p = math.log(sum(beams[prefix]) + 1e-15)
                candidates.append((text, acoustic_log_p))
                if len(candidates) >= top_k:
                    break

        return candidates

    def transcribe(self, audio_path: str):
        """Runs the complete two-pass pipeline on an audio file."""
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mel_transform(waveform).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_mel = log_mel.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.asr_model(log_mel)
            logits_2d = logits[0]

        # Pass 1: Generate Top-K Candidate Hypotheses
        candidates = self.get_top_k_candidates(logits_2d, top_k=4)
        if not candidates:
            # Fallback to argmax
            raw_ids = logits_2d.argmax(dim=-1).cpu().tolist()
            return ctc_decode(raw_ids, self.blank_id, self.sp), []

        # Pass 2: Rescore Candidates with Language Model
        rescored = []
        for text, acoustic_score in candidates:
            lm_score = self.compute_lm_score(text)
            combined_score = acoustic_score + (self.lm_weight * lm_score)
            rescored.append({
                "text": text,
                "acoustic_score": acoustic_score,
                "lm_score": lm_score,
                "total_score": combined_score
            })

        rescored.sort(key=lambda x: x["total_score"], reverse=True)
        best_text = rescored[0]["text"]
        return best_text, rescored


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_sample.wav"
    print("=" * 80)
    print("DEMONSTRATING TWO-PASS ASR PIPELINE (CONFORMER + LM RESCORING)")
    print("=" * 80)

    rescorer = TwoPassRescorer(
        asr_checkpoint_path=CONFORMER_CKPT,
        sp_model_path=TOKENIZER_MODEL,
        lm_model_id="gpt2",
        beam_width=8,
        lm_weight=0.4
    )

    if os.path.isfile(audio_file):
        best_hyp, candidates = rescorer.transcribe(audio_file)
        print(f"\n--> Audio File: {audio_file}")
        print("\n--- Pass 1 Candidate Hypotheses & Pass 2 Rescoring ---")
        for i, c in enumerate(candidates, 1):
            marker = " <-- [SELECTED BY LM]" if i == 1 else ""
            print(f"[{i}] \"{c['text'].upper()}\"")
            print(f"    Acoustic Score: {c['acoustic_score']:.2f} | LM Score: {c['lm_score']:.2f} | Total: {c['total_score']:.2f}{marker}")

        print("\n" + "=" * 60)
        print("FINAL TRANSCRIPTION OUTPUT:")
        print("=" * 60)
        print(best_hyp.upper())
