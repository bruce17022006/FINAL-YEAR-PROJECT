import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environment variables for E: drive caching
os.environ["HF_HOME"] = "E:/HF_Cache"
os.environ["TORCH_HOME"] = "E:/Torch_Cache"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Device: {device}")

# =====================================================================
# PATCH TORCHAUDIO AUDIO LOADING (Bypasses torchcodec bug on Windows)
# =====================================================================
def safe_load_audio(filepath):
    data, samplerate = sf.read(filepath)
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, samplerate

torchaudio.load = safe_load_audio

# =====================================================================
# UTILITY: EDIT DISTANCE CALCULATIONS
# =====================================================================
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for idx1, c1 in enumerate(s1):
        current_row = [idx1 + 1]
        for idx2, c2 in enumerate(s2):
            insertions = previous_row[idx2 + 1] + 1
            deletions = current_row[idx2] + 1
            substitutions = previous_row[idx2] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_normalized_edit_distance(str1: str, str2: str) -> float:
    s1 = str1.replace(" ", "").replace("_", "").replace("Ġ", "").lower()
    s2 = str2.replace(" ", "").replace("_", "").replace("Ġ", "").lower()
    raw_dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return raw_dist / max_len if max_len > 0 else 0.0

# =====================================================================
# MODULE 1: SEQUENCE-LEVEL ALIGNMENT (Dynamic Programming)
# =====================================================================
def sequence_level_alignment(lm_tokens: list, asr_tokens: list):
    """Computes 1-to-1 sequence alignment path using Dynamic Programming."""
    M, N = len(lm_tokens), len(asr_tokens)
    dp = [[0] * (N + 1) for _ in range(M + 1)]

    for i in range(M + 1):
        dp[i][0] = i
    for j in range(N + 1):
        dp[0][j] = j

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            dist = get_normalized_edit_distance(lm_tokens[i-1], asr_tokens[j-1])
            cost = 0 if dist <= 0.2 else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion
                dp[i][j-1] + 1,      # Insertion
                dp[i-1][j-1] + cost  # Substitution
            )

    i, j = M, N
    alignments = []
    while i > 0 and j > 0:
        dist = get_normalized_edit_distance(lm_tokens[i-1], asr_tokens[j-1])
        cost = 0 if dist <= 0.2 else 1
        
        if dp[i][j] == dp[i-1][j-1] + cost:
            if cost == 0:
                alignments.append((i-1, j-1))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            i -= 1
        else:
            j -= 1

    alignments.reverse()
    return alignments

# =====================================================================
# MODULE 2: CROSS-VOCABULARY KD LOSS MODULE
# =====================================================================
class CrossVocabularyKDLoss(nn.Module):
    def __init__(self, top_k=8, dist_threshold=0.2, temperature=3.0):
        super().__init__()
        self.top_k = top_k
        self.dist_threshold = dist_threshold
        self.temperature = temperature

    def forward(self, lm_logits, asr_logits, lm_tokens, asr_tokens, asr_vocab, llm_tokenizer):
        alignments = sequence_level_alignment(lm_tokens, asr_tokens)
        if not alignments:
            return torch.tensor(0.0, device=asr_logits.device, requires_grad=True)

        kd_loss = torch.tensor(0.0, device=asr_logits.device)
        valid_positions = 0

        for i_lm, j_asr in alignments:
            lm_logits_pos = lm_logits[i_lm] / self.temperature
            topk_vals, topk_indices = torch.topk(lm_logits_pos, k=self.top_k, dim=-1)
            topk_probs = F.softmax(topk_vals, dim=-1)

            asr_log_probs = F.log_softmax(asr_logits[j_asr] / self.temperature, dim=-1)
            mapped_teacher_dist = torch.zeros_like(asr_log_probs)
            has_valid_mapping = False

            for rank in range(self.top_k):
                token_id = topk_indices[rank].item()
                prob_val = topk_probs[rank].item()
                lm_token_str = llm_tokenizer.decode([token_id])

                best_asr_idx = None
                min_dist = float('inf')

                for asr_idx, asr_tok in enumerate(asr_vocab):
                    dist = get_normalized_edit_distance(lm_token_str, asr_tok)
                    if dist < min_dist:
                        min_dist = dist
                        best_asr_idx = asr_idx

                if min_dist <= self.dist_threshold and best_asr_idx is not None:
                    mapped_teacher_dist[best_asr_idx] += prob_val
                    has_valid_mapping = True

            if has_valid_mapping:
                mapped_teacher_dist = mapped_teacher_dist / (mapped_teacher_dist.sum() + 1e-8)
                kd_loss += -torch.sum(mapped_teacher_dist * asr_log_probs)
                valid_positions += 1

        if valid_positions > 0:
            kd_loss = (self.temperature ** 2) * (kd_loss / valid_positions)
        
        return kd_loss

# =====================================================================
# MODULE 3: STUDENT ASR MODEL
# =====================================================================
class SimpleASRModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, asr_vocab_size=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        )
        self.decoder_proj = nn.Linear(hidden_dim * 2, asr_vocab_size)

    def forward(self, speech_features):
        encoder_out, _ = self.encoder(speech_features)
        logits = self.decoder_proj(encoder_out)
        return logits

# =====================================================================
# MODULE 4: LOCAL DATASET PIPELINE & COLLATE
# =====================================================================
ASR_VOCAB = ["<pad>", "<unk>"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
char2idx = {c: i for i, c in enumerate(ASR_VOCAB)}

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)

def text_to_asr_tokens(text):
    return torch.tensor([char2idx.get(c, 1) for c in text.upper()])

def librispeech_collate_fn(batch):
    audio_features_list = []
    asr_tokens_list = []
    transcripts = []
    input_lengths = []
    target_lengths = []

    for waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id in batch:
        mel_spec = mel_transform(waveform).squeeze(0).transpose(0, 1)
        audio_features_list.append(mel_spec)
        input_lengths.append(mel_spec.shape[0])
        
        tokens = text_to_asr_tokens(transcript)
        asr_tokens_list.append(tokens)
        target_lengths.append(tokens.shape[0])
        transcripts.append(transcript)

    padded_audio = pad_sequence(audio_features_list, batch_first=True, padding_value=0.0)
    padded_asr = pad_sequence(asr_tokens_list, batch_first=True, padding_value=0)

    return {
        "speech_features": padded_audio,                      # [Batch, Frames, 80]
        "y_asr": padded_asr,                                 # [Batch, Seq_Len_ASR]
        "input_lengths": torch.tensor(input_lengths),        # Tensor of frame counts
        "target_lengths": torch.tensor(target_lengths),      # Tensor of token counts
        "raw_transcripts": transcripts
    }

# =====================================================================
# MODULE 5: GREEDY DECODER
# =====================================================================
def decode_asr_logits_to_text(asr_logits, asr_vocab):
    pred_indices = torch.argmax(asr_logits, dim=-1)
    decoded_sentences = []
    
    for batch_idx in range(pred_indices.shape[0]):
        indices = pred_indices[batch_idx].cpu().tolist()
        predicted_chars = []
        prev_idx = None
        
        for idx in indices:
            if idx != 0 and idx != prev_idx:
                if idx < len(asr_vocab):
                    predicted_chars.append(asr_vocab[idx])
            prev_idx = idx
            
        sentence = "".join(predicted_chars).strip()
        decoded_sentences.append(sentence)
        
    return decoded_sentences

# =====================================================================
# MODULE 6: EXECUTION & TRAINING LOOP
# =====================================================================
if __name__ == "__main__":
    print("\n--- Initializing Local Dataset Pipeline ---")
    
    DATA_ROOT = os.path.join("Data", "dev-clean")
    if not os.path.exists(DATA_ROOT):
        DATA_ROOT = os.path.join("Implementation", "Data", "dev-clean")
        
    dataset = torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="dev-clean", download=False)
    print(f"Total Local Audio Samples Loaded: {len(dataset)}")
    
    # Use batch_size = 1 for exact token sequence alignment
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=librispeech_collate_fn)
    
    # Setup Teacher LLM
    print("\n[Teacher Setup] Loading Pretrained LLM Tokenizer & Model...")
    llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
    llm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    llm_model.eval()
    
    student_asr = SimpleASRModel(asr_vocab_size=len(ASR_VOCAB)).to(device)
    optimizer = torch.optim.Adam(student_asr.parameters(), lr=1e-3)
    
    kd_loss_fn = CrossVocabularyKDLoss(top_k=8, dist_threshold=0.2, temperature=3.0)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    
    alpha = 0.1 # Distillation weight
    
    print("\n--- Executing Training & Overfitting Check ---")
    student_asr.train()
    
    # Fetch first batch to train and verify non-empty text generation
    batch = next(iter(train_loader))
    speech_features = batch["speech_features"].to(device)
    y_asr = batch["y_asr"].to(device)
    input_lengths = batch["input_lengths"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    raw_text = batch["raw_transcripts"][0]

    # Warmup loop to overfit on single batch and force non-empty CTC predictions
    print(f"Overfitting single batch to force CTC alignment convergence...")
    for step in range(1, 101):
        optimizer.zero_grad()
        
        # A. Student Forward
        asr_logits = student_asr(speech_features)
        
        # B. CTC Loss
        log_probs_ctc = F.log_softmax(asr_logits, dim=-1).transpose(0, 1)
        loss_asr = ctc_loss_fn(log_probs_ctc, y_asr, input_lengths, target_lengths)
        
        # C. Teacher Forward
        with torch.no_grad():
            inputs = llm_tokenizer(raw_text, return_tensors="pt").to(device)
            lm_logits = llm_model(**inputs).logits[0]
            
        # D. Prepare Token Sequences
        lm_token_ids = llm_tokenizer.encode(raw_text)
        lm_tokens = [llm_tokenizer.decode([tid]) for tid in lm_token_ids]
        asr_tokens = [c for c in raw_text.upper()]
        
        # E. Cross-KD Loss
        pooled_asr_logits = F.adaptive_avg_pool1d(asr_logits[0].transpose(0, 1), len(asr_tokens)).transpose(0, 1)
        loss_kd = kd_loss_fn(lm_logits, pooled_asr_logits, lm_tokens, asr_tokens, ASR_VOCAB, llm_tokenizer)
        
        # Total Loss & Backprop
        total_loss = (1 - alpha) * loss_asr + alpha * loss_kd
        total_loss.backward()
        optimizer.step()
        
        if step % 25 == 0:
            pred_sentence = decode_asr_logits_to_text(asr_logits, ASR_VOCAB)[0]
            print(f"Step [{step:3d}/100] | Total Loss: {total_loss.item():.4f} | CTC Loss: {loss_asr.item():.4f} | KD Loss: {loss_kd.item():.4f}")
            print(f"   -> Predicted ASR Output: '{pred_sentence}'")

    print("\n==================== VERIFICATION COMPLETE ====================")
    print(f"Ground Truth Text  : '{raw_text}'")
    print(f"Final ASR Output   : '{decode_asr_logits_to_text(asr_logits, ASR_VOCAB)[0]}'")
    print("=================================================================")