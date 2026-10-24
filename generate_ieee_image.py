import os
import math
from PIL import Image, ImageDraw, ImageFont

def generate_ieee_publication_image(output_path="ieee_architecture_diagram.png"):
    # 2x scale for 300 DPI crisp print rendering
    scale = 2
    w_base, h_base = 1560, 680
    width, height = w_base * scale, h_base * scale

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Load high-quality serif & sans-serif fonts
    font_path_serif = "C:/Windows/Fonts/times.ttf"
    font_path_serif_bd = "C:/Windows/Fonts/timesbd.ttf"
    font_path_serif_it = "C:/Windows/Fonts/timesi.ttf"
    font_path_sans_bd = "C:/Windows/Fonts/arialbd.ttf"
    font_path_sans = "C:/Windows/Fonts/arial.ttf"

    f_title = ImageFont.truetype(font_path_serif_bd, 18 * scale)
    f_sub = ImageFont.truetype(font_path_serif_it, 13 * scale)
    f_box_header = ImageFont.truetype(font_path_serif_bd, 12 * scale)
    f_box_body = ImageFont.truetype(font_path_serif, 11 * scale)
    f_box_small = ImageFont.truetype(font_path_serif, 10 * scale)
    f_badge = ImageFont.truetype(font_path_sans_bd, 9 * scale)
    f_caption = ImageFont.truetype(font_path_serif_it, 12 * scale)

    def S(v):
        return int(v * scale)

    # 1. Outer Container Border (Exact Template with IEEE polish)
    draw.rounded_rectangle(
        [S(15), S(15), S(1545), S(630)],
        radius=S(36),
        outline="#334155",
        width=S(2),
        fill="#ffffff"
    )

    # 2. Main Title Banner
    draw.text(
        (width // 2, S(48)),
        "Dynamic Knowledge Distillation Transfer for Multimodal Speech Recognition and Emotion Classification",
        fill="#0f172a",
        font=f_title,
        anchor="mm"
    )
    # Title underline
    draw.line([(S(380), S(60)), (S(1180), S(60))], fill="#cbd5e1", width=S(1))

    # Helper function to draw publication cards
    def draw_ieee_card(x, y, w, h, title_text, body_lines, fill_color, border_color, badge=None, badge_color="#2563eb", rounded=False, radius=6, left_align=False):
        box_rect = [S(x), S(y), S(x + w), S(y + h)]
        if rounded:
            draw.rounded_rectangle(box_rect, radius=S(radius), fill=fill_color, outline=border_color, width=S(1.5))
        else:
            draw.rectangle(box_rect, fill=fill_color, outline=border_color, width=S(1.5))

        cur_y = S(y) + S(8)

        # Header badge if present
        if badge:
            badge_h = S(16)
            b_box = [S(x), S(y), S(x + w), S(y) + badge_h]
            draw.rectangle(b_box, fill=badge_color)
            draw.text((S(x + w / 2), S(y) + badge_h // 2), badge, fill="#ffffff", font=f_badge, anchor="mm")
            cur_y += badge_h + S(4)

        if title_text:
            draw.text((S(x + w / 2), cur_y + S(6)), title_text, fill="#0f172a", font=f_box_header, anchor="mm")
            cur_y += S(18)

        line_h = S(15)
        for line in body_lines:
            if left_align:
                draw.text((S(x + 12), cur_y), line, fill="#1e293b", font=f_box_body)
            else:
                draw.text((S(x + w / 2), cur_y), line, fill="#1e293b", font=f_box_body, anchor="mt")
            cur_y += line_h

    # Helper for drawing clean orthogonal arrows
    def draw_ieee_arrow(p1, p2, label=None, label_color="#1e293b", pill_bg="#ffffff", pill_border="#cbd5e1", is_dashed=False):
        p1_s = (S(p1[0]), S(p1[1]))
        p2_s = (S(p2[0]), S(p2[1]))

        if is_dashed:
            # Draw dashed line
            dx = p2_s[0] - p1_s[0]
            dy = p2_s[1] - p1_s[1]
            dist = math.hypot(dx, dy)
            dash_len = S(8)
            gap_len = S(6)
            steps = int(dist // (dash_len + gap_len))
            ux, uy = dx / dist, dy / dist
            cx, cy = p1_s[0], p1_s[1]
            for _ in range(steps):
                nx = cx + ux * dash_len
                ny = cy + uy * dash_len
                draw.line([(cx, cy), (nx, ny)], fill=label_color, width=S(2))
                cx = nx + ux * gap_len
                cy = ny + uy * gap_len
            draw.line([(cx, cy), p2_s], fill=label_color, width=S(2))
        else:
            draw.line([p1_s, p2_s], fill=label_color, width=S(1.5))

        # Arrowhead
        dx = p2_s[0] - p1_s[0]
        dy = p2_s[1] - p1_s[1]
        if dx != 0 or dy != 0:
            angle = math.atan2(dy, dx)
            arr_len = S(8)
            a1 = angle + math.pi - 0.4
            a2 = angle + math.pi + 0.4
            tip_x, tip_y = p2_s[0], p2_s[1]
            x1 = tip_x + arr_len * math.cos(a1)
            y1 = tip_y + arr_len * math.sin(a1)
            x2 = tip_x + arr_len * math.cos(a2)
            y2 = tip_y + arr_len * math.sin(a2)
            draw.polygon([(tip_x, tip_y), (x1, y1), (x2, y2)], fill=label_color)

        if label:
            lx, ly = (p1_s[0] + p2_s[0]) // 2, (p1_s[1] + p2_s[1]) // 2
            bbox = draw.textbbox((lx, ly), label, font=f_badge, anchor="mm")
            pad_x, pad_y = S(6), S(3)
            pill = [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y]
            draw.rounded_rectangle(pill, radius=S(4), fill=pill_bg, outline=pill_border, width=S(1))
            draw.text((lx, ly), label, fill=label_color, font=f_badge, anchor="mm")

    # ==================== DRAWING THE CARDS ====================
    # 1. Dataset
    draw_ieee_card(45, 80, 130, 65, "Dataset", ["IEMOCAP and", "LibriSpeech"], "#f8fafc", "#64748b")

    # 2. Audio Wave
    draw_ieee_card(45, 205, 130, 60, "Audio Wave", ["16 kHz Mono WAV"], "#eff6ff", "#3b82f6")

    # 3. Input Transcript
    draw_ieee_card(45, 335, 130, 60, "Input Transcript", ["Reference Texts"], "#faf5ff", "#a855f7")

    # 4. Conformer Student Model
    draw_ieee_card(
        255, 195, 160, 85, "Conformer Backbone",
        ["Encoder-Decoder Model", "(4x Subsampling, 6L)", "θ_S (Trainable)"],
        "#eff6ff", "#1d4ed8", badge="STUDENT ASR MODEL", badge_color="#1d4ed8"
    )

    # 5. Dialogue Segmentation
    draw_ieee_card(255, 335, 160, 60, "", ["1) Dialogue Segmentation", "2) Human Annotation"], "#faf5ff", "#9333ea")

    # 6. LoRA LLaMA (Rounded)
    draw_ieee_card(
        470, 330, 155, 70, "LLaMA-3.2-1B Model",
        ["LoRA Adaptation", "(2048-dim Embeddings)"],
        "#faf5ff", "#7e22ce", badge="FROZEN TEACHER", badge_color="#7e22ce", rounded=True, radius=12
    )

    # 7. Pseudo Bidirectional
    draw_ieee_card(470, 455, 155, 60, "Pseudo Bidirectional", ["Approach", "Sequence Length L' = 8"], "#faf5ff", "#a855f7")

    # 8. Sinkhorn Optimal Transport Plan
    draw_ieee_card(
        685, 140, 180, 135, "Optimal Transport Plan",
        [
            "Many-to-Many Token Map",
            "1) Ground Cost Matrix C",
            "2) Top-K Selection",
            "Entropic Regularization ε"
        ],
        "#fffbeb", "#d97706", badge="CORE DISTILLATION NOVELTY", badge_color="#d97706"
    )

    # 9. ASR Result
    draw_ieee_card(930, 160, 140, 60, "ASR Result", ["Predicted Tokens (Ŷ)"], "#eff6ff", "#2563eb")

    # 10. Emotion Head
    draw_ieee_card(
        930, 260, 140, 60, "Emotion Classifier",
        ["Temporal Pooling + MLP", "4-Class (IEMOCAP)"],
        "#ecfdf5", "#059669", badge="SER HEAD", badge_color="#059669"
    )

    # 11. Wasserstein Loss
    draw_ieee_card(685, 385, 180, 55, "Wasserstein OT Loss", ["L_OT = <P*, C> + ε H(P*)"], "#fffbeb", "#b45309")

    # 12. Total Loss L
    draw_ieee_card(
        930, 385, 140, 65, "Total Joint Loss L",
        ["L_CTC + 0.1·L_OT + 0.2·L_e"],
        "#fff1f2", "#e11d48", badge="MULTI-TASK OBJECTIVE", badge_color="#e11d48"
    )

    # 13. Evaluation Metrics
    draw_ieee_card(
        1130, 160, 155, 170, "",
        [
            "Evaluation Metrics",
            "",
            "ASR Criteria:",
            "  1) CER (%)",
            "  2) WER (%)",
            "  3) CTC Loss",
            "SER Criteria:",
            "  4) Weighted Acc (WA)",
            "  5) Unweighted Acc (UA)"
        ],
        "#f8fafc", "#475569", badge="EVALUATION METRICS", badge_color="#475569", left_align=True
    )

    # 14. System Output
    draw_ieee_card(
        1345, 195, 150, 105, "",
        [
            "System Outputs",
            "",
            "1) Recognized",
            "   Speech Text",
            "2) Emotion Label",
            "   (Ang/Hap/Sad/Neu)"
        ],
        "#eff6ff", "#2563eb", badge="PREDICTIONS", badge_color="#2563eb", left_align=True
    )

    # ==================== DRAWING THE PATHWAYS ====================
    draw_ieee_arrow((110, 145), (110, 205))  # Dataset -> Audio
    draw_ieee_arrow((110, 265), (110, 335))  # Audio -> Transcript

    # Audio -> Conformer
    draw_ieee_arrow((175, 235), (255, 235), label="Raw Audio", label_color="#1e40af", pill_bg="#eff6ff", pill_border="#bfdbfe")

    # Transcript -> Dialogue Segmentation
    draw_ieee_arrow((175, 365), (255, 365), label="Raw Text", label_color="#6b21a8", pill_bg="#faf5ff", pill_border="#e9d5ff")

    # Dialogue -> LLaMA
    draw_ieee_arrow((415, 365), (470, 365))

    # Pseudo-bidirectional -> LLaMA
    draw_ieee_arrow((547, 455), (547, 400))

    # Conformer -> Sinkhorn Plan
    draw_ieee_arrow((415, 215), (685, 215), label="Audio Embeddings", label_color="#1d4ed8", pill_bg="#eff6ff", pill_border="#bfdbfe")

    # LLaMA -> Sinkhorn Plan
    p_llama = [(S(625), S(365)), (S(655), S(365)), (S(655), S(240)), (S(685), S(240))]
    draw.line(p_llama, fill="#1e293b", width=S(1.5))
    draw_ieee_arrow((655, 240), (685, 240))
    # text label
    draw_ieee_arrow((625, 365), (655, 365), label="Text Embeds", label_color="#6b21a8", pill_bg="#faf5ff", pill_border="#e9d5ff")

    # Conformer -> Emotion Head
    p_emo = [(S(415), S(250)), (S(445), S(250)), (S(445), S(290)), (S(930), S(290))]
    draw.line(p_emo, fill="#059669", width=S(1.5))
    draw_ieee_arrow((445, 290), (930, 290), label_color="#059669")

    # Sinkhorn -> ASR Result
    draw_ieee_arrow((865, 190), (930, 190))

    # Sinkhorn -> Wasserstein Loss
    draw_ieee_arrow((775, 275), (775, 385))

    # ASR Result -> Total Loss
    draw_ieee_arrow((1000, 220), (1000, 385))

    # Emotion Head -> Total Loss
    draw_ieee_arrow((1040, 320), (1040, 385), label_color="#059669")

    # Wasserstein Loss -> Total Loss
    draw_ieee_arrow((865, 415), (930, 415))

    # Backpropagation Loop: Total Loss -> Conformer
    p_backprop = [(S(930), S(435)), (S(335), S(435)), (S(335), S(280))]
    draw_ieee_arrow((930, 435), (335, 435), label="Updation of ASR Weights Only", label_color="#be123c", pill_bg="#fff1f2", pill_border="#fecdd3", is_dashed=True)
    draw_ieee_arrow((335, 435), (335, 280), label_color="#be123c", is_dashed=True)

    # Total Loss -> Evaluation Metrics
    p_eval = [(S(1070), S(415)), (S(1207), S(415)), (S(1207), S(330))]
    draw_ieee_arrow((1070, 415), (1207, 415), label="Minimize Loss", label_color="#334155", pill_bg="#f8fafc", pill_border="#cbd5e1")
    draw_ieee_arrow((1207, 415), (1207, 330))

    # Evaluation Metrics -> Output
    draw_ieee_arrow((1285, 245), (1345, 245))

    # 3. Figure Caption at Bottom (Standard IEEE Two-Column Format)
    draw.text(
        (width // 2, S(655)),
        "Fig. 1. Architectural schematic of the proposed Dynamic Knowledge Distillation Transfer framework. The student Conformer is guided via Sinkhorn Optimal Transport from frozen LLaMA-3.2-1B semantic priors to optimize both ASR and Emotion Classification.",
        fill="#334155",
        font=f_caption,
        anchor="mm"
    )

    img.save(output_path, dpi=(300, 300))
    print(f"--> High-DPI IEEE Publication Image generated at: {output_path}")

if __name__ == "__main__":
    generate_ieee_publication_image()
