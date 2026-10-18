import os
import glob
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# 1. Configurations & Paths
# ==========================================

DATA_DIR = os.path.join(
    "Data", "dev-clean", "LibriSpeech", "dev-clean"
)

OUTPUT_DIR = "precomputed_pbkd_dev_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "meta-llama/Llama-3.2-1B"

# ------------------------------------------
# PBKD Parameters
# ------------------------------------------

K_PRIME = 32          # Candidate tokens
L_PRIME = 3           # Lookahead length
FINAL_TOP_K = 8       # Final teacher tokens
TEMPERATURE = 3.0

# Small batch for 4 GB VRAM
BATCH_CHUNK_SIZE = 2

# Maximum previous context
MAX_PREFIX_LEN = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)

print(f"--> Using Device: {DEVICE}")
print(f"--> Data Directory: {DATA_DIR}")
print(f"--> Output Directory: {OUTPUT_DIR}")


# ==========================================
# 2. Load Tokenizer + Teacher Model
# ==========================================

print("\n[1/4] Loading Pretrained Teacher LLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    low_cpu_mem_usage=True
)

model = model.to(DEVICE)

model.eval()

# Important for memory
model.config.use_cache = False

print("--> Teacher Model Loaded Successfully!")


# ==========================================
# 3. Save LLM Embedding Table
# ==========================================

print("\n[2/4] Saving LLM Input Embedding Matrix...")

embed_save_path = os.path.join(
    OUTPUT_DIR,
    "llm_embeddings.pt"
)

if not os.path.exists(embed_save_path):

    with torch.no_grad():

        llm_embeds = (
            model
            .get_input_embeddings()
            .weight
            .detach()
            .cpu()
        )

        torch.save(
            llm_embeds,
            embed_save_path
        )

        print(
            f"Saved Embeddings: "
            f"{llm_embeds.shape}"
        )

else:

    print(
        f"--> Embeddings already exist at "
        f"{embed_save_path}, skipping."
    )


# ==========================================
# 4. Parse LibriSpeech Transcripts
# ==========================================

def load_librispeech_transcripts(root_dir):

    trans_files = glob.glob(
        os.path.join(
            root_dir,
            "**",
            "*.trans.txt"
        ),
        recursive=True
    )

    dataset = {}

    for tf in trans_files:

        with open(
            tf,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) >= 2:

                    utt_id = parts[0]

                    text = " ".join(
                        parts[1:]
                    ).lower()

                    dataset[utt_id] = text

    return dataset


print(
    "\n[3/4] Parsing LibriSpeech "
    "*.trans.txt files..."
)

transcripts = load_librispeech_transcripts(
    DATA_DIR
)

print(
    f"Found {len(transcripts)} "
    f"utterances to process."
)


# ==========================================
# 5. GPU Memory Helper
# ==========================================

def gpu_memory():

    if DEVICE != "cuda":
        return ""

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    return (
        f" | VRAM "
        f"{allocated:.2f}/"
        f"{reserved:.2f} GB"
    )


# ==========================================
# 6. PBKD Computation
# ==========================================

@torch.inference_mode()
def compute_pbkd_for_sequence(token_ids):

    seq_len = len(token_ids)

    topk_indices_list = []
    topk_probs_list = []

    # ------------------------------------------
    # IMPORTANT:
    # Do NOT keep all vocabulary logits.
    #
    # Instead, process the sequence in small
    # pieces to reduce GPU memory.
    # ------------------------------------------

    for i in range(seq_len):

        # ======================================
        # Step A: Obtain causal distribution
        # ======================================

        # Context before current token
        context_start = max(
            0,
            i - MAX_PREFIX_LEN
        )

        context = token_ids[
            context_start:i + 1
        ]

        context = context.unsqueeze(0).to(
            DEVICE
        )

        # Forward pass
        outputs = model(
            context,
            use_cache=False
        )

        # Last position predicts next token
        logits = outputs.logits[:, -1, :]

        # Convert to FP32 for numerical stability
        logits = logits.float()

        probs = F.softmax(
            logits,
            dim=-1
        )

        # Candidate tokens
        k = min(
            K_PRIME,
            probs.size(-1)
        )

        top_kprime_probs, top_kprime_indices = (
            torch.topk(
                probs,
                k=k,
                dim=-1
            )
        )

        top_kprime_probs = (
            top_kprime_probs.squeeze(0)
        )

        top_kprime_indices = (
            top_kprime_indices.squeeze(0)
        )

        # Cleanup
        del outputs
        del logits
        del probs
        del context

        # ======================================
        # Step B: PBKD Lookahead
        # ======================================

        future_len = min(
            L_PRIME,
            seq_len - (i + 1)
        )

        if future_len > 0:

            future_tokens = token_ids[
                i + 1:
                i + 1 + future_len
            ].to(DEVICE)

            prefix_tokens = token_ids[
                max(0, i - MAX_PREFIX_LEN):
                i
            ].to(DEVICE)

            start_idx = len(prefix_tokens)

            candidate_scores = []

            # ----------------------------------
            # Candidate batches
            # ----------------------------------

            for start in range(
                0,
                k,
                BATCH_CHUNK_SIZE
            ):

                end = min(
                    start + BATCH_CHUNK_SIZE,
                    k
                )

                candidate_ids = (
                    top_kprime_indices[
                        start:end
                    ]
                )

                candidate_sequences = []

                for candidate_id in candidate_ids:

                    sequence = torch.cat(
                        [
                            prefix_tokens,
                            candidate_id.view(1),
                            future_tokens
                        ]
                    )

                    candidate_sequences.append(
                        sequence
                    )

                candidate_batch = torch.stack(
                    candidate_sequences
                ).to(DEVICE)

                # ----------------------------------
                # Teacher forward pass
                # ----------------------------------

                candidate_outputs = model(
                    candidate_batch,
                    use_cache=False
                )

                candidate_logits = (
                    candidate_outputs.logits
                )

                batch_scores = torch.zeros(
                    candidate_batch.size(0),
                    device=DEVICE,
                    dtype=torch.float32
                )

                # ----------------------------------
                # Calculate future probability
                # ----------------------------------

                for step in range(future_len):

                    position = (
                        start_idx + step
                    )

                    step_logits = (
                        candidate_logits[
                            :,
                            position,
                            :
                        ].float()
                    )

                    target_token = (
                        future_tokens[step]
                    )

                    target_logit = (
                        step_logits[
                            :,
                            target_token
                        ]
                    )

                    log_probability = (
                        target_logit
                        - torch.logsumexp(
                            step_logits,
                            dim=-1
                        )
                    )

                    batch_scores += (
                        log_probability
                    )

                    del step_logits
                    del target_logit
                    del log_probability

                candidate_scores.append(
                    batch_scores.cpu()
                )

                # Cleanup
                del candidate_outputs
                del candidate_logits
                del candidate_batch
                del candidate_sequences
                del candidate_ids
                del batch_scores

            # ----------------------------------
            # Combine candidate scores
            # ----------------------------------

            future_log_probs = torch.cat(
                candidate_scores
            ).to(DEVICE)

            log_prior = torch.log(
                top_kprime_probs + 1e-12
            )

            combined_log_prob = (
                log_prior +
                future_log_probs
            )

            pb_probs = F.softmax(
                combined_log_prob,
                dim=-1
            )

            # Cleanup
            del future_log_probs
            del candidate_scores
            del log_prior
            del combined_log_prob
            del future_tokens
            del prefix_tokens

        else:

            pb_probs = (
                top_kprime_probs /
                top_kprime_probs.sum()
            )

        # ======================================
        # Step C: Final Top-K
        # ======================================

        final_k = min(
            FINAL_TOP_K,
            pb_probs.numel()
        )

        final_probs, rank_idx = torch.topk(
            pb_probs,
            k=final_k
        )

        selected_ids = (
            top_kprime_indices[
                rank_idx
            ]
        )

        # ======================================
        # Step D: Temperature Scaling
        # ======================================

        scaled_logits = (
            torch.log(
                final_probs + 1e-12
            )
            / TEMPERATURE
        )

        scaled_probs = F.softmax(
            scaled_logits,
            dim=-1
        )

        # ======================================
        # Save CPU results
        # ======================================

        topk_indices_list.append(
            selected_ids.cpu()
        )

        topk_probs_list.append(
            scaled_probs.cpu()
        )

        # ======================================
        # Cleanup
        # ======================================

        del top_kprime_probs
        del top_kprime_indices
        del pb_probs
        del final_probs
        del rank_idx
        del selected_ids
        del scaled_logits
        del scaled_probs

        if DEVICE == "cuda":

            torch.cuda.empty_cache()

    # ==========================================
    # Return final tensors
    # ==========================================

    return (
        torch.stack(
            topk_indices_list
        ),
        torch.stack(
            topk_probs_list
        )
    )


# ==========================================
# 7. Precomputation Loop
# ==========================================

print(
    "\n[4/4] Computing & Saving "
    "PBKD Teacher Labels..."
)

for utt_id, text in tqdm(
    transcripts.items(),
    desc="Processing Transcripts"
):

    save_file = os.path.join(
        OUTPUT_DIR,
        f"{utt_id}.pt"
    )

    # ------------------------------------------
    # Resume capability
    # ------------------------------------------

    if os.path.exists(save_file):

        continue

    # ------------------------------------------
    # Tokenize
    # ------------------------------------------

    token_ids = tokenizer.encode(
        text,
        return_tensors="pt"
    ).squeeze(0)

    # ------------------------------------------
    # PBKD
    # ------------------------------------------

    topk_indices, topk_probs = (
        compute_pbkd_for_sequence(
            token_ids
        )
    )

    # ------------------------------------------
    # Save
    # ------------------------------------------

    torch.save(
        {
            "utt_id": utt_id,
            "token_ids": token_ids,
            "topk_indices": topk_indices,
            "topk_probs": topk_probs
        },
        save_file
    )

    # ------------------------------------------
    # Cleanup
    # ------------------------------------------

    del token_ids
    del topk_indices
    del topk_probs

    if DEVICE == "cuda":

        torch.cuda.empty_cache()


print(
    "\nStage 1 Offline Extraction "
    "Completed Successfully!"
)