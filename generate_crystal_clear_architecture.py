import os
from PIL import Image, ImageDraw, ImageFont

font_serif_bd = "C:/Windows/Fonts/timesbd.ttf"
font_serif = "C:/Windows/Fonts/times.ttf"
font_sans_bd = "C:/Windows/Fonts/arialbd.ttf"
font_sans = "C:/Windows/Fonts/arial.ttf"

def generate_crystal_clear_architecture(output_path="ieee_architecture_diagram.png"):
    # 3x scaling factor for ultra-sharp 300 DPI print clarity
    scale = 3
    def S(v):
        return int(v * scale)

    # Tight, balanced canvas: 1150 width x 560 height (zero wasted margins!)
    w, h = S(1150), S(560)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    # High-contrast bold typography (readable at single-column scaling!)
    f_box_title = ImageFont.truetype(font_sans_bd, S(13))
    f_box_sub = ImageFont.truetype(font_sans, S(11))
    f_box_bold = ImageFont.truetype(font_sans_bd, S(11))
    f_badge = ImageFont.truetype(font_sans_bd, S(10))
    f_arrow = ImageFont.truetype(font_sans_bd, S(10.5))

    def draw_bold_arrow(p1, p2, col="#1e293b", width=S(2.5)):
        draw.line([p1, p2], fill=col, width=width)
        # Arrowhead
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if abs(dx) > abs(dy):
            # Horizontal arrow
            sign = 1 if dx > 0 else -1
            draw.polygon([
                (p2[0], p2[1]),
                (p2[0] - sign * S(9), p2[1] - S(6)),
                (p2[0] - sign * S(9), p2[1] + S(6))
            ], fill=col)
        else:
            # Vertical arrow
            sign = 1 if dy > 0 else -1
            draw.polygon([
                (p2[0], p2[1]),
                (p2[0] - S(6), p2[1] - sign * S(9)),
                (p2[0] + S(6), p2[1] - sign * S(9))
            ], fill=col)

    # =========================================================================
    # 1. INPUT STAGE (LEFT COLUMN)
    # =========================================================================
    # Audio Waveform Box
    bx_audio = (S(25), S(50), S(190), S(140))
    draw.rounded_rectangle(bx_audio, radius=S(8), fill="#f0f9ff", outline="#0284c7", width=S(2))
    draw.text((S(107), S(75)), "Raw Audio Wave", fill="#0369a1", font=f_box_title, anchor="mm")
    draw.text((S(107), S(100)), "16 kHz Mono WAV", fill="#334155", font=f_box_sub, anchor="mm")
    draw.text((S(107), S(120)), "(LibriSpeech / IEMOCAP)", fill="#64748b", font=f_box_sub, anchor="mm")

    # Front-end Filterbank Box
    bx_mel = (S(25), S(190), S(190), S(280))
    draw.rounded_rectangle(bx_mel, radius=S(8), fill="#f8fafc", outline="#475569", width=S(2))
    draw.text((S(107), S(215)), "Mel Spectrogram", fill="#1e293b", font=f_box_title, anchor="mm")
    draw.text((S(107), S(240)), "80-Channel Log-Mel", fill="#334155", font=f_box_bold, anchor="mm")
    draw.text((S(107), S(260)), "Delta + Delta-Delta", fill="#64748b", font=f_box_sub, anchor="mm")

    draw_bold_arrow((S(107), S(140)), (S(107), S(190)), col="#0284c7")

    # Ground Truth Text Box (Bottom Left)
    bx_text = (S(25), S(420), S(190), S(510))
    draw.rounded_rectangle(bx_text, radius=S(8), fill="#faf5ff", outline="#9333ea", width=S(2))
    draw.text((S(107), S(445)), "Reference Text", fill="#7e22ce", font=f_box_title, anchor="mm")
    draw.text((S(107), S(470)), "Dialogue Transcripts", fill="#334155", font=f_box_sub, anchor="mm")
    draw.text((S(107), S(490)), "BPE Tokenization", fill="#64748b", font=f_box_sub, anchor="mm")

    # =========================================================================
    # 2. STUDENT CONFORMER BACKBONE (CENTER-LEFT COLUMN)
    # =========================================================================
    bx_student = (S(245), S(50), S(510), S(340))
    draw.rounded_rectangle(bx_student, radius=S(12), fill="#ffffff", outline="#2563eb", width=S(2.5))
    
    # Header Banner for Student
    draw.rounded_rectangle([S(245), S(50), S(510), S(92)], radius=S(10), fill="#2563eb")
    draw.text((S(377), S(71)), "STUDENT CONFORMER ASR", fill="#ffffff", font=f_box_title, anchor="mm")

    draw.text((S(377), S(115)), "4x 2D-Conv Subsampling", fill="#0f172a", font=f_box_bold, anchor="mm")
    draw.text((S(377), S(135)), "Stride (2,2) x 2 Layers", fill="#64748b", font=f_box_sub, anchor="mm")

    draw.line([(S(270), S(155)), (S(485), S(155))], fill="#e2e8f0", width=S(1.5))

    draw.text((S(377), S(180)), "6-Layer Conformer Encoder", fill="#1e293b", font=f_box_title, anchor="mm")
    draw.text((S(377), S(205)), "Encoder Dim: d = 256  |  4 Heads", fill="#334155", font=f_box_bold, anchor="mm")
    draw.text((S(377), S(230)), "Macaron Feed-Forward (Swish)", fill="#475569", font=f_box_sub, anchor="mm")
    draw.text((S(377), S(252)), "Depthwise Conv1D (Kernel K=15)", fill="#475569", font=f_box_sub, anchor="mm")
    draw.text((S(377), S(274)), "Relative Positional Encoding", fill="#475569", font=f_box_sub, anchor="mm")

    # Parameter Badge
    draw.rounded_rectangle([S(265), S(295), S(490), S(328)], radius=S(6), fill="#dbeafe", outline="#3b82f6")
    draw.text((S(377), S(311)), "Total Capacity: 12.55M Parameters", fill="#1d4ed8", font=f_badge, anchor="mm")

    # Arrow from Mel to Student
    draw_bold_arrow((S(190), S(235)), (S(245), S(235)), col="#2563eb")

    # =========================================================================
    # 3. FROZEN CAUSAL LLM TEACHER (BOTTOM-CENTER)
    # =========================================================================
    bx_teacher = (S(245), S(410), S(510), S(520))
    draw.rounded_rectangle(bx_teacher, radius=S(10), fill="#faf5ff", outline="#7e22ce", width=S(2.5))
    
    draw.rounded_rectangle([S(245), S(410), S(510), S(448)], radius=S(8), fill="#7e22ce")
    draw.text((S(377), S(429)), "FROZEN CAUSAL TEACHER", fill="#ffffff", font=f_box_title, anchor="mm")

    draw.text((S(377), S(465)), "meta-llama / Llama-3.2-1B", fill="#581c87", font=f_box_title, anchor="mm")
    draw.text((S(377), S(488)), "Hidden State: d_teacher = 2048", fill="#334155", font=f_box_bold, anchor="mm")
    draw.text((S(377), S(507)), "Vocab: 128,256 BPE  |  16 Heads", fill="#64748b", font=f_box_sub, anchor="mm")

    draw_bold_arrow((S(190), S(465)), (S(245), S(465)), col="#7e22ce")

    # =========================================================================
    # 4. CORE NOVELTY: SINKHORN-WASSERSTEIN OPTIMAL TRANSPORT (CENTER)
    # =========================================================================
    bx_ot = (S(565), S(260), S(840), S(440))
    draw.rounded_rectangle(bx_ot, radius=S(10), fill="#fffbeb", outline="#d97706", width=S(2.5))

    draw.rounded_rectangle([S(565), S(260), S(840), S(298)], radius=S(8), fill="#d97706")
    draw.text((S(702), S(279)), "CROSS-VOCABULARY SINKHORN OT", fill="#ffffff", font=f_box_title, anchor="mm")

    draw.text((S(702), S(318)), "Linear Projection: 256 -> 2048", fill="#92400e", font=f_box_bold, anchor="mm")
    draw.text((S(702), S(340)), "Cosine Cost Matrix: C = 1 - cos(s', t)", fill="#1e293b", font=f_box_sub, anchor="mm")
    draw.text((S(702), S(362)), "Gibbs Kernel: K = exp(-C / epsilon)", fill="#1e293b", font=f_box_sub, anchor="mm")
    draw.text((S(702), S(384)), "Sinkhorn Recursions: L = 10 Steps", fill="#1e293b", font=f_box_sub, anchor="mm")

    draw.rounded_rectangle([S(580), S(402), S(825), S(430)], radius=S(5), fill="#fef3c7", outline="#f59e0b")
    draw.text((S(702), S(416)), "Continuous Mass Transfer (Zero Discard)", fill="#b45309", font=f_badge, anchor="mm")

    # Connect Student to OT (from bottom of student)
    draw.line([(S(470), S(340)), (S(470), S(350)), (S(565), S(350))], fill="#2563eb", width=S(2.5))
    draw_bold_arrow((S(520), S(350)), (S(565), S(350)), col="#2563eb")
    draw.text((S(510), S(338)), "d=256", fill="#2563eb", font=f_arrow, anchor="mm")

    # Connect Teacher to OT (from top of teacher)
    draw.line([(S(470), S(410)), (S(470), S(380)), (S(565), S(380))], fill="#7e22ce", width=S(2.5))
    draw_bold_arrow((S(520), S(380)), (S(565), S(380)), col="#7e22ce")
    draw.text((S(510), S(392)), "d=2048", fill="#7e22ce", font=f_arrow, anchor="mm")

    # =========================================================================
    # 5. DUAL EXECUTION HEADS (RIGHT COLUMN)
    # =========================================================================
    # Head 1: CTC ASR Head
    bx_ctc = (S(565), S(50), S(840), S(140))
    draw.rounded_rectangle(bx_ctc, radius=S(8), fill="#f0f9ff", outline="#0284c7", width=S(2))
    draw.text((S(702), S(75)), "ASR Transcription Head", fill="#0369a1", font=f_box_title, anchor="mm")
    draw.text((S(702), S(98)), "Linear CTC Layer (256 -> 1024)", fill="#334155", font=f_box_bold, anchor="mm")
    draw.text((S(702), S(120)), "Greedy CTC Decode + Blank Penalty", fill="#64748b", font=f_box_sub, anchor="mm")

    # Head 2: SER Emotion Head
    bx_ser = (S(565), S(155), S(840), S(245))
    draw.rounded_rectangle(bx_ser, radius=S(8), fill="#f0fdf4", outline="#16a34a", width=S(2))
    draw.text((S(702), S(180)), "Speech Emotion Head (SER)", fill="#15803d", font=f_box_title, anchor="mm")
    draw.text((S(702), S(203)), "Temporal Average Pooling + MLP", fill="#334155", font=f_box_bold, anchor="mm")
    draw.text((S(702), S(225)), "4-Class Classifier (Neutral, Anger, Sad, Happy)", fill="#64748b", font=f_box_sub, anchor="mm")

    # Connect Student to CTC Head
    draw.line([(S(510), S(95)), (S(565), S(95))], fill="#0284c7", width=S(2.5))
    draw_bold_arrow((S(530), S(95)), (S(565), S(95)), col="#0284c7")

    # Connect Student to SER Head
    draw.line([(S(510), S(200)), (S(565), S(200))], fill="#16a34a", width=S(2.5))
    draw_bold_arrow((S(530), S(200)), (S(565), S(200)), col="#16a34a")

    # =========================================================================
    # 6. COMPOSITE MULTI-TASK LOSS & FINAL OUTPUTS (FAR RIGHT)
    # =========================================================================
    bx_loss = (S(890), S(290), S(1125), S(420))
    draw.rounded_rectangle(bx_loss, radius=S(8), fill="#fef2f2", outline="#dc2626", width=S(2))
    draw.text((S(1007), S(315)), "MULTI-TASK OBJECTIVE", fill="#991b1b", font=f_box_title, anchor="mm")
    draw.text((S(1007), S(342)), "L_total = L_CTC + 0.5 L_emo + 0.1 L_OT", fill="#1e293b", font=f_box_bold, anchor="mm")
    draw.text((S(1007), S(368)), "Simultaneous Backpropagation", fill="#475569", font=f_box_sub, anchor="mm")
    draw.text((S(1007), S(395)), "Student Gradient Updates Only", fill="#dc2626", font=f_badge, anchor="mm")

    # Connect OT to Multi-Task Loss
    draw_bold_arrow((S(840), S(355)), (S(890), S(355)), col="#d97706")

    # Final Output Results Box
    bx_out = (S(890), S(50), S(1125), S(245))
    draw.rounded_rectangle(bx_out, radius=S(8), fill="#1e293b", outline="#0f172a", width=S(2))
    draw.text((S(1007), S(75)), "EVALUATED OUTPUTS", fill="#38bdf8", font=f_box_title, anchor="mm")
    draw.text((S(1007), S(105)), "1. Recognized Speech Text", fill="#ffffff", font=f_box_bold, anchor="mm")
    draw.text((S(1007), S(126)), "(37.5% WER, 23.4% CER)", fill="#94a3b8", font=f_box_sub, anchor="mm")
    draw.text((S(1007), S(156)), "2. Predicted Emotion Class", fill="#ffffff", font=f_box_bold, anchor="mm")
    draw.text((S(1007), S(177)), "(59.97% UA on Held-Out Ses-5)", fill="#94a3b8", font=f_box_sub, anchor="mm")
    draw.text((S(1007), S(206)), "3. Ambiguity Resolution", fill="#34d399", font=f_box_bold, anchor="mm")
    draw.text((S(1007), S(227)), "(72.72% Human Consensus)", fill="#a7f3d0", font=f_box_sub, anchor="mm")

    # Connect CTC to Output
    draw_bold_arrow((S(840), S(95)), (S(890), S(95)), col="#0284c7")
    # Connect SER to Output
    draw_bold_arrow((S(840), S(200)), (S(890), S(200)), col="#16a34a")

    img.save(output_path, dpi=(300, 300))
    print(f"Generated Crystal Clear Figure 1: {output_path}")

if __name__ == "__main__":
    generate_crystal_clear_architecture()
