import os
from PIL import Image, ImageDraw, ImageFont

font_serif_bd = "C:/Windows/Fonts/timesbd.ttf"
font_serif = "C:/Windows/Fonts/times.ttf"
font_serif_it = "C:/Windows/Fonts/timesi.ttf"
font_sans_bd = "C:/Windows/Fonts/arialbd.ttf"
font_sans = "C:/Windows/Fonts/arial.ttf"

def S(v, scale=2):
    return int(v * scale)

def generate_exact_base_paper_style_fig2(output_path="fig2_cross_vocab_alignment.png"):
    scale = 2
    w_base, h_base = 1200, 680
    w, h = w_base * scale, h_base * scale

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    f_title = ImageFont.truetype(font_serif_bd, 13 * scale)
    f_sub = ImageFont.truetype(font_serif_bd, 11 * scale)
    f_txt = ImageFont.truetype(font_serif, 10 * scale)
    f_txt_bd = ImageFont.truetype(font_serif_bd, 10 * scale)
    f_math = ImageFont.truetype(font_serif_it, 10 * scale)
    f_math_bd = ImageFont.truetype(font_serif_bd, 11 * scale)
    f_sm = ImageFont.truetype(font_serif, 9 * scale)
    f_sm_it = ImageFont.truetype(font_serif_it, 9 * scale)

    # Dividing line in center (Clean academic separator)
    draw.line([(S(600), S(20)), (S(600), S(660))], fill="#1f2937", width=S(1))

    # =========================================================================
    # LEFT PANEL: (a) Conventional Cross-Vocabulary KD (Futami & Kawahara 2026)
    # EXACT REPLICA OF BASE PAPER FIGURE 2
    # =========================================================================
    draw.text((S(300), S(25)), "(a) Conventional Cross-Vocabulary KD (Futami et al., 2026)", fill="#000000", font=f_title, anchor="mm")
    
    # 1) Sequence-level alignment header
    draw.text((S(40), S(55)), "(1) Sequence-level alignment", fill="#000000", font=f_sub)

    # Table y^LM
    draw.text((S(45), S(95)), "y", fill="#000000", font=f_math_bd)
    draw.text((S(53), S(90)), "LM", fill="#000000", font=f_sm)
    draw.text((S(68), S(95)), ":", fill="#000000", font=f_txt_bd)

    # Table y^ASR
    draw.text((S(45), S(165)), "y", fill="#000000", font=f_math_bd)
    draw.text((S(53), S(160)), "ASR", fill="#000000", font=f_sm)
    draw.text((S(72), S(165)), ":", fill="#000000", font=f_txt_bd)

    # Coordinates for token grids
    gx = S(85)
    gy_lm = S(80)
    gy_asr = S(150)
    th = S(34)

    # y^LM columns: [observing (w=130)], [empty (w=40)], [_my (w=60)], [_anxiety (w=80)], [_to (w=55)], [_understand (w=110)]
    lm_cols = [
        ("1\nobserving", S(135)),
        ("", S(45)),
        ("2\n_my", S(65)),
        ("3\n_anxiety", S(85)),
        ("4\n_to", S(60)),
        ("5\n_understand", S(105))
    ]

    cx = gx
    for label, cw in lm_cols:
        draw.rectangle([cx, gy_lm, cx + cw, gy_lm + th], outline="#000000", fill="#ffffff", width=S(1))
        if label:
            draw.text((cx + S(5), gy_lm + S(4)), label.split('\n')[0], fill="#333333", font=f_sm)
            draw.text((cx + S(16), gy_lm + S(16)), label.split('\n')[1], fill="#000000", font=f_txt)
        cx += cw

    # y^ASR columns: [obs (w=65)], [erving (w=70)], [_my (w=65)], [_anxiety (w=85)], [_to (w=60)], [_understand (w=105)]
    asr_cols = [
        ("1\nobs", S(65)),
        ("2\nerving", S(70)),
        ("3\n_my", S(65)),
        ("4\n_anxiety", S(85)),
        ("5\n_to", S(60)),
        ("6\n_understand", S(105))
    ]

    cx = gx
    for label, cw in asr_cols:
        draw.rectangle([cx, gy_asr, cx + cw, gy_asr + th], outline="#000000", fill="#ffffff", width=S(1))
        draw.text((cx + S(5), gy_asr + S(4)), label.split('\n')[0], fill="#333333", font=f_sm)
        draw.text((cx + S(16), gy_asr + S(16)), label.split('\n')[1], fill="#000000", font=f_txt)
        cx += cw

    # Draw Red Arrows pointing down for matches
    def draw_down_arrow(x, y1, y2):
        draw.line([(x, y1), (x, y2)], fill="#dc2626", width=S(2))
        draw.polygon([(x - S(4), y2 - S(5)), (x + S(4), y2 - S(5)), (x, y2 + S(2))], fill="#dc2626")

    # Arrow for _my
    draw_down_arrow(gx + S(180) + S(32), gy_lm + th, gy_asr)
    # Arrow for _anxiety
    draw_down_arrow(gx + S(245) + S(42), gy_lm + th, gy_asr)
    # Arrow for _to
    draw_down_arrow(gx + S(330) + S(30), gy_lm + th, gy_asr)
    # Arrow for _understand
    draw_down_arrow(gx + S(390) + S(52), gy_lm + th, gy_asr)

    # Red Box highlighting _anxiety match (exact replica of user's uploaded image!)
    anx_x1 = gx + S(245)
    draw.rectangle([anx_x1 - S(2), gy_lm - S(2), anx_x1 + S(87), gy_asr + th + S(2)], outline="#dc2626", width=S(2))

    # RED X for observing -> obs + erving (THE DISCARD MECHANISM)
    obs_x_mid = gx + S(68)
    draw.text((obs_x_mid, S(125)), "DISCARDED (X)", fill="#dc2626", font=f_sm_it, anchor="mm")
    draw.line([(gx + S(30), S(120)), (gx + S(105), S(135))], fill="#dc2626", width=S(1))
    draw.line([(gx + S(30), S(135)), (gx + S(105), S(120))], fill="#dc2626", width=S(1))

    # (2) Distribution-level alignment box
    draw.rounded_rectangle([S(40), S(220), S(570), S(530)], radius=S(4), outline="#000000", width=S(1))
    draw.text((S(55), S(235)), "(2) Distribution-level alignment (1-to-1 Mapping)", fill="#000000", font=f_sub)

    # Arrow from red box to distribution box
    draw.line([(anx_x1 + S(42), gy_asr + th + S(2)), (anx_x1 + S(42), S(220))], fill="#dc2626", width=S(2))
    draw.polygon([(anx_x1 + S(38), S(216)), (anx_x1 + S(46), S(216)), (anx_x1 + S(42), S(223))], fill="#dc2626")

    # Table P_3^LM
    draw.text((S(105), S(270)), "P", fill="#000000", font=f_math_bd)
    draw.text((S(114), S(265)), "LM", fill="#000000", font=f_sm)
    draw.text((S(114), S(275)), "3", fill="#000000", font=f_sm)
    draw.text((S(132), S(270)), ":", fill="#000000", font=f_txt_bd)

    # Table P_4^ASR
    draw.text((S(345), S(270)), "P", fill="#000000", font=f_math_bd)
    draw.text((S(354), S(265)), "ASR", fill="#000000", font=f_sm)
    draw.text((S(354), S(275)), "4", fill="#000000", font=f_sm)
    draw.text((S(374), S(270)), ":", fill="#000000", font=f_txt_bd)

    # Draw distribution tables
    lm_dist = [("_struggle", "0.21"), ("_life", "0.14"), ("_failure", "0.13"), ("_attempts", "0.12")]
    asr_dist = ["_struggle", "_life", "_failure", "_attempt"]

    dt_y0 = S(290)
    dt_h = S(28)
    for i, ((tok, prob), asr_tok) in enumerate(zip(lm_dist, asr_dist)):
        # Left LM table
        draw.rectangle([S(85), dt_y0 + i*dt_h, S(190), dt_y0 + (i+1)*dt_h], outline="#000000", fill="#ffffff", width=S(1))
        draw.text((S(92), dt_y0 + i*dt_h + S(6)), tok, fill="#000000", font=f_txt)
        draw.rectangle([S(190), dt_y0 + i*dt_h, S(250), dt_y0 + (i+1)*dt_h], outline="#000000", fill="#ffffff", width=S(1))
        draw.text((S(202), dt_y0 + i*dt_h + S(6)), prob, fill="#000000", font=f_txt)

        # Red block arrow
        draw.polygon([
            (S(270), dt_y0 + i*dt_h + S(8)),
            (S(310), dt_y0 + i*dt_h + S(8)),
            (S(310), dt_y0 + i*dt_h + S(4)),
            (S(325), dt_y0 + i*dt_h + S(14)),
            (S(310), dt_y0 + i*dt_h + S(24)),
            (S(310), dt_y0 + i*dt_h + S(20)),
            (S(270), dt_y0 + i*dt_h + S(20))
        ], fill="#dc2626", outline="#991b1b")

        # Right ASR table
        draw.rectangle([S(345), dt_y0 + i*dt_h, S(460), dt_y0 + (i+1)*dt_h], outline="#000000", fill="#ffffff", width=S(1))
        draw.text((S(355), dt_y0 + i*dt_h + S(6)), asr_tok, fill="#000000", font=f_txt)

    # Limitation caption at bottom of Panel A
    draw.rectangle([S(40), S(545), S(570), S(650)], outline="#94a3b8", fill="#f9fafb", width=S(1))
    draw.text((S(50), S(555)), "Base Paper Limitation:", fill="#b91c1c", font=f_txt_bd)
    draw.text((S(50), S(578)), "• Enforces strictly 1-to-1 token alignment based on edit distance (d <= 0.2).", fill="#333333", font=f_sm)
    draw.text((S(50), S(600)), "• If a word splits into multiple subwords (e.g. 'observing' -> 'obs' + 'erving'),", fill="#333333", font=f_sm)
    draw.text((S(50), S(622)), "  no alignment is established and KD is COMPLETELY DISCARDED.", fill="#b91c1c", font=f_sm)


    # =========================================================================
    # RIGHT PANEL: (b) Proposed Continuous Sinkhorn Optimal Transport (Ours)
    # ACADEMIC FORMULATION WITH SOFT TRANSPORT PLAN MATRIX P*
    # =========================================================================
    draw.text((S(900), S(25)), "(b) Proposed Sinkhorn-Wasserstein Optimal Transport (Ours)", fill="#000000", font=f_title, anchor="mm")

    # 1) Sequence-level continuous coupling
    draw.text((S(640), S(55)), "(1) Continuous Sequence-Level Mass Transport (Zero Discard)", fill="#000000", font=f_sub)

    # Table y^LM
    draw.text((S(645), S(95)), "y", fill="#000000", font=f_math_bd)
    draw.text((S(653), S(90)), "LM", fill="#000000", font=f_sm)
    draw.text((S(668), S(95)), ":", fill="#000000", font=f_txt_bd)

    # Table y^ASR
    draw.text((S(645), S(165)), "y", fill="#000000", font=f_math_bd)
    draw.text((S(653), S(160)), "ASR", fill="#000000", font=f_sm)
    draw.text((S(672), S(165)), ":", fill="#000000", font=f_txt_bd)

    rgx = S(685)
    cx = rgx
    for label, cw in lm_cols:
        draw.rectangle([cx, gy_lm, cx + cw, gy_lm + th], outline="#000000", fill="#ffffff", width=S(1))
        if label:
            draw.text((cx + S(5), gy_lm + S(4)), label.split('\n')[0], fill="#333333", font=f_sm)
            draw.text((cx + S(16), gy_lm + S(16)), label.split('\n')[1], fill="#000000", font=f_txt)
        cx += cw

    cx = rgx
    for label, cw in asr_cols:
        draw.rectangle([cx, gy_asr, cx + cw, gy_asr + th], outline="#000000", fill="#ffffff", width=S(1))
        draw.text((cx + S(5), gy_asr + S(4)), label.split('\n')[0], fill="#333333", font=f_sm)
        draw.text((cx + S(16), gy_asr + S(16)), label.split('\n')[1], fill="#000000", font=f_txt)
        cx += cw

    # In our method: GREEN CONNECTOR WITH WEIGHTS FOR MULTI-TOKEN SPLITS!
    # observing -> obs (weight 0.62)
    draw.line([(rgx + S(68), gy_lm + th), (rgx + S(32), gy_asr)], fill="#15803d", width=S(2))
    draw.text((rgx + S(38), S(125)), "P*=0.62", fill="#15803d", font=f_sm, anchor="mm")

    # observing -> erving (weight 0.38)
    draw.line([(rgx + S(68), gy_lm + th), (rgx + S(100), gy_asr)], fill="#15803d", width=S(2))
    draw.text((rgx + S(102), S(125)), "P*=0.38", fill="#15803d", font=f_sm, anchor="mm")

    # Connectors for other tokens
    draw_down_arrow(rgx + S(180) + S(32), gy_lm + th, gy_asr)
    draw_down_arrow(rgx + S(245) + S(42), gy_lm + th, gy_asr)
    draw_down_arrow(rgx + S(330) + S(30), gy_lm + th, gy_asr)
    draw_down_arrow(rgx + S(390) + S(52), gy_lm + th, gy_asr)

    # (2) Continuous Distribution-level Optimal Transport Plan
    draw.rounded_rectangle([S(640), S(220), S(1170), S(530)], radius=S(4), outline="#000000", width=S(1))
    draw.text((S(655), S(235)), "(2) Entropic Optimal Transport Coupling Matrix P*", fill="#000000", font=f_sub)

    # Draw Matrix P* with cell values
    mx0 = S(760)
    my0 = S(275)
    mcw = S(75)
    mch = S(32)

    ot_matrix = [
        ["0.62", "0.03", "0.01", "0.00", "0.00"],
        ["0.38", "0.04", "0.01", "0.00", "0.00"],
        ["0.00", "0.91", "0.02", "0.01", "0.00"],
        ["0.00", "0.01", "0.94", "0.01", "0.00"],
        ["0.00", "0.00", "0.01", "0.96", "0.01"],
        ["0.00", "0.01", "0.01", "0.02", "0.99"]
    ]

    col_headers = ["observing", "_my", "_anxiety", "_to", "_understand"]
    row_headers = ["obs", "erving", "_my", "_anxiety", "_to", "_understand"]

    # Draw column headers
    for j, ch_tok in enumerate(col_headers):
        draw.text((mx0 + j*mcw + mcw//2, my0 - S(10)), ch_tok, fill="#000000", font=f_sm, anchor="mm")

    # Draw row headers and matrix cells
    for i, rh_tok in enumerate(row_headers):
        draw.text((mx0 - S(15), my0 + i*mch + mch//2), rh_tok, fill="#000000", font=f_sm, anchor="rm")
        for j in range(5):
            val = ot_matrix[i][j]
            # light green highlight for active mass
            fill_c = "#dcfce7" if float(val) > 0.3 else "#ffffff"
            draw.rectangle([mx0 + j*mcw, my0 + i*mch, mx0 + (j+1)*mcw, my0 + (i+1)*mch], outline="#666666", fill=fill_c, width=1)
            draw.text((mx0 + j*mcw + mcw//2, my0 + i*mch + mch//2), val, fill="#000000", font=f_txt, anchor="mm")

    # Mathematical formulation footer in Panel B
    draw.rectangle([S(640), S(545), S(1170), S(650)], outline="#16a34a", fill="#f0fdf4", width=S(1))
    draw.text((S(650), S(555)), "Proposed Optimal Transport Contribution:", fill="#15803d", font=f_txt_bd)
    draw.text((S(650), S(578)), "• Formulates cross-vocabulary transfer via Gibbs kernel K = exp(-C / eps).", fill="#1f2937", font=f_sm)
    draw.text((S(650), S(600)), "• When subwords fragment ('observing' -> 'obs' + 'erving'), continuous mass", fill="#1f2937", font=f_sm)
    draw.text((S(650), S(622)), "  conservation holds: sum_j P*_{ij} = r_i. ZERO TOKENS DISCARDED.", fill="#15803d", font=f_txt_bd)

    img.save(output_path, dpi=(300, 300))
    print(f"Generated Base-Paper Style Figure 2: {output_path}")

if __name__ == "__main__":
    generate_exact_base_paper_style_fig2()
