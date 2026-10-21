import math
import torch
import torch.nn.functional as F
import sentencepiece as spm
from collections import defaultdict

class WordTrie:
    def __init__(self):
        self.root = {}
        self.word_end = "__END__"

    def insert(self, word):
        node = self.root
        for char in word.lower():
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.word_end] = True

    def is_valid_prefix(self, prefix):
        node = self.root
        for char in prefix.lower():
            if char not in node:
                return False
            node = node[char]
        return True

    def is_valid_word(self, word):
        node = self.root
        for char in word.lower():
            if char not in node:
                return False
            node = node[char]
        return self.word_end in node


class CTCBeamSearchDecoder:
    def __init__(
        self,
        sp_model_path: str,
        lexicon_words: set = None,
        beam_width: int = 8,
        blank_penalty: float = 0.4,
        word_bonus: float = 0.25,
        prune_threshold: float = 0.005,
    ):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.blank_id = self.sp.piece_to_id("<blank>")
        self.pad_id = self.sp.pad_id()
        self.beam_width = beam_width
        self.blank_penalty = blank_penalty
        self.word_bonus = word_bonus
        self.prune_threshold = prune_threshold

        self.trie = WordTrie()
        if lexicon_words:
            for w in lexicon_words:
                self.trie.insert(w)
        self.has_lexicon = bool(lexicon_words)

    def decode(self, logits: torch.Tensor) -> str:
        T, V = logits.shape
        logits = logits.clone()
        if self.blank_penalty > 0.0:
            logits[:, self.blank_id] -= self.blank_penalty

        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()

        # Beams map: tuple_of_token_ids -> (p_blank, p_non_blank)
        beams = {(): (1.0, 0.0)}

        for t in range(T):
            step_probs = probs[t]
            top_indices = [
                i for i, p in enumerate(step_probs) if p > self.prune_threshold or i == self.blank_id
            ]

            next_beams = defaultdict(lambda: (0.0, 0.0))

            for prefix, (p_b, p_nb) in beams.items():
                p_total = p_b + p_nb
                if p_total <= 0:
                    continue

                # 1. Blank emission
                blank_prob = step_probs[self.blank_id]
                nb_b, nb_nb = next_beams[prefix]
                next_beams[prefix] = (nb_b + p_total * blank_prob, nb_nb)

                # 2. Non-blank emissions
                last_token = prefix[-1] if len(prefix) > 0 else None

                for c in top_indices:
                    if c == self.blank_id or c == self.pad_id:
                        continue

                    prob_c = step_probs[c]

                    if c == last_token:
                        new_prefix = prefix + (c,)
                        nb_b_new, nb_nb_new = next_beams[new_prefix]
                        next_beams[new_prefix] = (nb_b_new, nb_nb_new + p_b * prob_c)

                        nb_b_cur, nb_nb_cur = next_beams[prefix]
                        next_beams[prefix] = (nb_b_cur, nb_nb_cur + p_nb * prob_c)
                    else:
                        new_prefix = prefix + (c,)
                        nb_b_new, nb_nb_new = next_beams[new_prefix]
                        next_beams[new_prefix] = (nb_b_new, nb_nb_new + p_total * prob_c)

            # Prune to beam_width with scoring
            scored_beams = []
            for prefix, (p_b, p_nb) in next_beams.items():
                tot_p = p_b + p_nb
                if tot_p <= 0.0:
                    continue

                log_score = math.log(tot_p + 1e-15)

                if self.has_lexicon and len(prefix) > 0:
                    piece_str = self.sp.id_to_piece(prefix[-1])
                    # Only score at word boundaries (\u2581 indicates space)
                    if piece_str.startswith("\u2581"):
                        decoded_text = self.sp.decode_ids(list(prefix)).strip()
                        words = decoded_text.split()
                        if len(words) >= 2:
                            prev_word = words[-2]
                            if self.trie.is_valid_word(prev_word):
                                log_score += self.word_bonus

                scored_beams.append((log_score, prefix, (p_b, p_nb)))

            scored_beams.sort(key=lambda x: x[0], reverse=True)
            beams = {prefix: probs_tuple for _, prefix, probs_tuple in scored_beams[: self.beam_width]}

        if not beams:
            return ""

        best_prefix = max(beams.keys(), key=lambda p: math.log(sum(beams[p]) + 1e-15))
        return self.sp.decode_ids(list(best_prefix)).strip()
