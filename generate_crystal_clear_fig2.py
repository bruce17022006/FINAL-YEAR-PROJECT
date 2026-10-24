import os
from PIL import Image, ImageDraw, ImageFont

font_serif_bd = "C:/Windows/Fonts/timesbd.ttf"
font_serif = "C:/Windows/Fonts/times.ttf"
font_serif_it = "C:/Windows/Fonts/timesi.ttf"
font_sans_bd = "C:/Windows/Fonts/arialbd.ttf"
font_sans = "C:/Windows/Fonts/arial.ttf"

def generate_crystal_clear_fig2(output_path="fig2_cross_vocab_alignment.png"):
    # 2.5x scale for razor-sharp 300 DPI print rendering
    scale = 3
    def S(v):
        return int(v * scale)

    # Tight canvas: 1200 width x 490 height (eliminates all dead whitespace!)
    w, h = S(1200), S(490)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    # High-contrast bold typography for crystal clarity in PDF
    f_title = ImageFont.truetype(font_serif_bd, S(14))
    f_sec = ImageFont.truetype(font_serif_bd, S(12))
    f_tok = ImageFont.truetype(font_serif_bd, S(12))
    f_tok_reg = ImageFont.truetype(font_serif, S(12))
    f_num = ImageFont.truetype(font_serif_bd, S(10))
    f_math = ImageFont.truetype(font_serif_it, S(12))
    f_math_bd = ImageFont.truetype(font_serif_bd, S(13))
    f_sm_bd = ImageFont.truetype(font_serif_bd, S(10))
    f_matrix = ImageFont.truetype(font_serif_bd, S(11))

    # Center dividing vertical line
    draw.line([(S(600), S(10)), (S(600), S(480))], fill="#000000", width=S(1.5))

    # =========================================================================
    # LEFT PANEL: (a) Conventional Cross-Vocabulary KD (Futami & Kawahara 2026)
    # =========================================================================
    draw.text((S(300), S(22)), "(a) Conventional Cross-Vocabulary KD (Futami et al., 2026)", fill="#000000", font=f_title, anchor="mm")
    draw.text((S(25), S(48)), "(1) Sequence-level alignment (Rigid 1-to-1)", fill="#000000", font=f_sec)

    # Labels y^LM and y^ASR
    draw.text((S(25), S(82)), "y", fill="#000000", font=f_math_bd)
    draw.text((S(34), S(76)), "LM", fill="#000000", font=f_sm_bd)
    draw.text((S(52), S(82)), ":", fill="#000000", font=f_title)

    draw.text((S(25), S(142)), "y", fill="#000000", font=f_math_bd)
    draw.text((S(34), S(136)), "ASR", fill="#000000", font=f_sm_bd)
    draw.text((S(56), S(142)), ":", fill="#000000", font=f_title)

    gx = S(68)
    gy_lm = S(70)
    gy_asr = S(130)
    th = S(32)

    # y^LM columns: [observing (130)], [empty (40)], [_my (65)], [_anxiety (85)], [_to (55)], [_understand (110)]
    lm_widths = [S(130), S(40), S(65), S(85), S(55), S(110)]
    lm_labels = [("1", "observing"), ("", ""), ("2", "_my"), ("3", "_anxiety"), ("4", "_to"), ("5", "_understand")]

    cx = gx
    for (num, tok), cw in zip(lm_labels, lm_widths):
        draw.rectangle([cx, gy_lm, cx + cw, gy_lm + th], outline="#000000", fill="#ffffff", width=S(1.5))
        if num:
            draw.text((cx + S(4), gy_lm + S(3)), num, fill="#333333", font=f_num)
            draw.text((cx + S(16), gy_lm + S(15)), tok, fill="#000000", font=f_tok)
        cx += cw

    # y^ASR columns: [obs (65)], [erving (65)], [_my (65)], [_anxiety (85)], [_to (55)], [_understand (110)]
    asr_widths = [S(65), S(65), S(65), S(85), S(55), S(110)]
    asr_labels = [("1", "obs"), ("2", "erving"), ("3", "_my"), ("4", "_anxiety"), ("5", "_to"), ("6", "_understand")]

    cx = gx
    for (num, tok), cw in zip(asr_labels, asr_widths):
        draw.rectangle([cx, gy_asr, cx + cw, gy_asr + th], outline="#000000", fill="#ffffff", width=S(1.5))
        draw.text((cx + S(4), gy_asr + S(3)), num, fill="#333333", font=f_num)
        draw.text((cx + S(16), gy_asr + S(15)), tok, fill="#000000", font=f_tok)
        cx += cw

    # Clean downward arrows
    def draw_bold_arrow(x, y1, y2, col="#dc2626"):
        draw.line([(x, y1), (x, y2)], fill=col, width=S(2))
        draw.polygon([(x - S(4.5), y2 - S(6)), (x + S(4.5), y2 - S(6)), (x, y2 + S(1))], fill=col)

    # 1-to-1 arrows
    draw_bold_arrow(gx + S(170) + S(32), gy_lm + th, gy_asr)
    draw_bold_arrow(gx + S(235) + S(42), gy_lm + th, gy_asr)
    draw_bold_arrow(gx + S(320) + S(27), gy_lm + th, gy_asr)
    draw_bold_arrow(gx + S(375) + S(55), gy_lm + th, gy_asr)

    # Bold RED outline around _anxiety match (clean & sharp!)
    anx_x0 = gx + S(235)
    anx_w = S(85)
    draw.rectangle([anx_x0 - S(2), gy_lm - S(2), anx_x0 + anx_w + S(2), gy_asr + th + S(2)], outline="#dc2626", width=S(2.5))

    # RED X for observing -> obs + erving
    draw.line([(gx + S(35), S(110)), (gx + S(95), S(122))], fill="#dc2626", width=S(1.5))
    draw.line([(gx + S(35), S(122)), (gx + S(95), S(110))], fill="#dc2626", width=S(1.5))
    draw.text((gx + S(65), S(116)), "DISCARDED (X)", fill="#dc2626", font=f_sm_bd, anchor="mm")

    # (2) Distribution-level alignment Box
    d_box_y0 = S(195)
    d_box_h = S(275)
    draw.rounded_rectangle([S(20), d_box_y0, S(580), d_box_y0 + d_box_h], radius=S(4), outline="#000000", width=S(1.5))
    draw.text((S(35), d_box_y0 + S(14)), "(2) Distribution-level alignment (1-to-1 Mapping)", fill="#000000", font=f_sec)

    # Arrow from red box to distribution box
    draw.line([(anx_x0 + anx_w//2, gy_asr + th + S(2)), (anx_x0 + anx_w//2, d_box_y0)], fill="#dc2626", width=S(2.5))
    draw.polygon([(anx_x0 + anx_w//2 - S(5), d_box_y0 - S(6)), (anx_x0 + anx_w//2 + S(5), d_box_y0 - S(6)), (anx_x0 + anx_w//2, d_box_y0 + S(2))], fill="#dc2626")

    # Headers P_3^LM and P_4^ASR
    draw.text((S(110), d_box_y0 + S(46)), "P", fill="#000000", font=f_math_bd)
    draw.text((S(120), d_box_y0 + S(40)), "LM", fill="#000000", font=f_sm_bd)
    draw.text((S(120), d_box_y0 + S(52)), "3", fill="#000000", font=f_sm_bd)
    draw.text((S(142), d_box_y0 + S(46)), ":", fill="#000000", font=f_title)

    draw.text((S(360), d_box_y0 + S(46)), "P", fill="#000000", font=f_math_bd)
    draw.text((S(370), d_box_y0 + S(40)), "ASR", fill="#000000", font=f_sm_bd)
    draw.text((S(370), d_box_y0 + S(52)), "4", fill="#000000", font=f_sm_bd)
    draw.text((S(396), d_box_y0 + S(46)), ":", fill="#000000", font=f_title)

    lm_dist = [("_struggle", "0.21"), ("_life", "0.14"), ("_failure", "0.13"), ("_attempts", "0.12")]
    asr_dist = ["_struggle", "_life", "_failure", "_attempt"]

    row_y0 = d_box_y0 + S(68)
    row_h = S(28)
    for i, ((tok, prob), asr_tok) in enumerate(zip(lm_dist, asr_dist)):
        ry = row_y0 + i * row_h
        # LM Token cell
        draw.rectangle([S(75), ry, S(185), ry + row_h], outline="#000000", fill="#ffffff", width=S(1.2))
        draw.text((S(83), ry + S(6)), tok, fill="#000000", font=f_tok)

        # LM Prob cell
        draw.rectangle([S(185), ry, S(250), ry + row_h], outline="#000000", fill="#ffffff", width=S(1.2))
        draw.text((S(200), ry + S(6)), prob, fill="#000000", font=f_tok)

        # Red block arrow
        draw.polygon([
            (S(275), ry + S(9)),
            (S(315), ry + S(9)),
            (S(315), ry + S(4)),
            (S(335), ry + S(14)),
            (S(315), ry + S(24)),
            (S(315), ry + S(19)),
            (S(275), ry + S(19))
        ], fill="#dc2626", outline="#991b1b")

        # ASR Token cell
        draw.rectangle([S(360), ry, S(480), ry + row_h], outline="#000000", fill="#ffffff", width=S(1.2))
        draw.text((S(370), ry + S(6)), asr_tok, fill="#000000", font=f_tok)

    # Footnote inside Panel A
    draw.text((S(45), d_box_y0 + S(200)), "Note: Subwords like 'observing' -> 'obs'+'erving' have no 1-to-1 match", fill="#b91c1c", font=f_sm_bd)
    draw.text((S(45), d_box_y0 + S(218)), "and are COMPLETELY DISCARDED from distillation.", fill="#b91c1c", font=f_sm_bd)


    # =========================================================================
    # RIGHT PANEL: (b) Proposed Continuous Sinkhorn Optimal Transport (Ours)
    # =========================================================================
    draw.text((S(900), S(22)), "(b) Proposed Sinkhorn-Wasserstein Optimal Transport (Ours)", fill="#000000", font=f_title, anchor="mm")
    draw.text((S(625), S(48)), "(1) Continuous Sequence-Level Transport (Zero Discard)", fill="#000000", font=f_sec)

    # Labels y^LM and y^ASR
    draw.text((S(625), S(82)), "y", fill="#000000", font=f_math_bd)
    draw.text((S(634), S(76)), "LM", fill="#000000", font=f_sm_bd)
    draw.text((S(652), S(82)), ":", fill="#000000", font=f_title)

    draw.text((S(625), S(142)), "y", fill="#000000", font=f_math_bd)
    draw.text((S(634), S(136)), "ASR", fill="#000000", font=f_sm_bd)
    draw.text((S(656), S(142)), ":", fill="#000000", font=f_title)

    rgx = S(668)
    cx = rgx
    for (num, tok), cw in zip(lm_labels, lm_widths):
        draw.rectangle([cx, gy_lm, cx + cw, gy_lm + th], outline="#000000", fill="#ffffff", width=S(1.5))
        if num:
            draw.text((cx + S(4), gy_lm + S(3)), num, fill="#333333", font=f_num)
            draw.text((cx + S(16), gy_lm + S(15)), tok, fill="#000000", font=f_tok)
        cx += cw

    cx = rgx
    for (num, tok), cw in zip(asr_labels, asr_widths):
        draw.rectangle([cx, gy_asr, cx + cw, gy_asr + th], outline="#000000", fill="#ffffff", width=S(1.5))
        draw.text((cx + S(4), gy_asr + S(3)), num, fill="#333333", font=f_num)
        draw.text((cx + S(16), gy_asr + S(15)), tok, fill="#000000", font=f_tok)
        cx += cw

    # Bold Green Connections for Multi-Token Subwords! (Conserved probability mass!)
    draw.line([(rgx + S(65), gy_lm + th), (rgx + S(32), gy_asr)], fill="#15803d", width=S(2.5))
    draw.text((rgx + S(35), S(116)), "P*=0.62", fill="#15803d", font=f_sm_bd, anchor="mm")

    draw.line([(rgx + S(65), gy_lm + th), (rgx + S(98), gy_asr)], fill="#15803d", width=S(2.5))
    draw.text((rgx + S(105), S(116)), "P*=0.38", fill="#15803d", font=f_sm_bd, anchor="mm")

    # Regular arrows
    draw_bold_arrow(rgx + S(170) + S(32), gy_lm + th, gy_asr)
    draw_bold_arrow(rgx + S(235) + S(42), gy_lm + th, gy_asr)
    draw_bold_arrow(rgx + S(320) + S(27), gy_lm + th, gy_asr)
    draw_bold_arrow(rgx + S(375) + S(55), gy_lm + th, gy_asr)

    # (2) Continuous Optimal Transport Matrix Box
    draw.rounded_rectangle([S(620), d_box_y0, S(1180), d_box_y0 + d_box_h], radius=S(4), outline="#000000", width=S(1.5))
    draw.text((S(635), d_box_y0 + S(14)), "(2) Entropic Optimal Transport Coupling Matrix P*", fill="#000000", font=f_sec)

    # Matrix P* values
    mx0 = S(755)
    my0 = d_box_y0 + S(55)
    mcw = S(78)
    mch = S(27)

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
        draw.text((mx0 + j*mcw + mcw//2, my0 - S(10)), ch_tok, fill="#000000", font=f_tok_reg, anchor="mm")

    # Draw row headers and matrix cells
    for i, rh_tok in enumerate(row_headers):
        draw.text((mx0 - S(12), my0 + i*mch + mch//2), rh_tok, fill="#000000", font=f_tok_reg, anchor="rm")
        for j in range(5):
            val = ot_matrix[i][j]
            bg_col = "#dcfce7" if float(val) > 0.3 else "#ffffff"
            draw.rectangle([mx0 + j*mcw, my0 + i*mch, mx0 + (j+1)*mcw, my0 + (i+1)*mch], outline="#000000", fill=bg_col, width=S(1))
            draw.text((mx0 + j*mcw + mcw//2, my0 + i*mch + mch//2), val, fill="#000000", font=f_matrix, anchor="mm")

    # Footnote inside Panel B
    draw.text((S(645), d_box_y0 + S(235)), "Continuous mass conservation holds: sum_j P*_{ij} = r_i.", fill="#15803d", font=f_sm_bd)
    draw.text((S(645), d_box_y0 + S(252)), "ZERO TOKENS DISCARDED across disparate subword vocabularies.", fill="#15803d", font=f_sm_bd)

    img.save(output_path, dpi=(300, 300))
    print(f"Generated Crystal Clear Figure 2: {output_path}")

if __name__ == "__main__":
    generate_crystal_clear_fig2()
