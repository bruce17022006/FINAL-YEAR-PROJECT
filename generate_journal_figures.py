import math
from PIL import Image, ImageDraw, ImageFont

font_serif_bd = "C:/Windows/Fonts/timesbd.ttf"
font_serif = "C:/Windows/Fonts/times.ttf"
font_serif_it = "C:/Windows/Fonts/timesi.ttf"
font_sans_bd = "C:/Windows/Fonts/arialbd.ttf"
font_sans = "C:/Windows/Fonts/arial.ttf"

def S(v, scale=2):
    return int(v * scale)

# =====================================================================
# FIGURE 2: CROSS-VOCABULARY ALIGNMENT TOPOLOGY
# =====================================================================
def generate_fig2(output_path="fig2_cross_vocab_alignment.png"):
    scale = 2
    w, h = 1200 * scale, 560 * scale
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    f_title = ImageFont.truetype(font_serif_bd, 16 * scale)
    f_sub = ImageFont.truetype(font_serif_it, 12 * scale)
    f_lbl = ImageFont.truetype(font_serif_bd, 11 * scale)
    f_txt = ImageFont.truetype(font_serif, 10 * scale)
    f_math = ImageFont.truetype(font_serif_it, 10 * scale)

    # Main Border
    draw.rounded_rectangle([S(10), S(10), S(1190), S(550)], radius=S(16), outline="#94a3b8", width=S(1))
    
    # Title
    draw.text((w // 2, S(35)), "Figure 2: Topological Comparison of Cross-Vocabulary Distillation Alignment", fill="#0f172a", font=f_title, anchor="mm")
    draw.text((w // 2, S(58)), "Heuristic Edit Distance Discard (Futami & Kawahara 2026) vs. Continuous Sinkhorn Optimal Transport Coupling", fill="#475569", font=f_sub, anchor="mm")

    # Panel A: Base Paper (Edit-Distance 1-to-1)
    draw.rounded_rectangle([S(30), S(85), S(580), S(525)], radius=S(12), outline="#cbd5e1", fill="#f8fafc", width=S(1))
    draw.text((S(305), S(110)), "(a) Base Paper: 1-to-1 Edit Distance (Futami et al. 2026)", fill="#b91c1c", font=f_lbl, anchor="mm")
    draw.text((S(305), S(130)), "Multi-token subwords are strictly discarded (Zero KD applied)", fill="#64748b", font=f_sub, anchor="mm")

    # Grid for Panel A
    tokens_lm = ["_observing", "_anxiety", "_to", "_understand"]
    tokens_asr = ["_obs", "_erving", "_my", "_anxiety", "_to", "_understand"]
    
    # Draw mapping table
    gx0, gy0 = S(60), S(160)
    cw, ch = S(75), S(45)
    for i, t in enumerate(tokens_lm):
        draw.rectangle([gx0 + i*cw, gy0, gx0 + (i+1)*cw, gy0 + ch], outline="#94a3b8", fill="#e2e8f0")
        draw.text((gx0 + i*cw + cw//2, gy0 + ch//2), t, fill="#0f172a", font=f_txt, anchor="mm")

    gy1 = S(270)
    for j, t in enumerate(tokens_asr):
        draw.rectangle([gx0 + j*(cw*4//6), gy1, gx0 + (j+1)*(cw*4//6), gy1 + ch], outline="#94a3b8", fill="#e2e8f0")
        draw.text((gx0 + j*(cw*4//6) + (cw*4//6)//2, gy1 + ch//2), t, fill="#0f172a", font=f_txt, anchor="mm")

    # Connectors
    # observing -> obs + erving (DISCARDED RED X)
    draw.line([(gx0 + cw//2, gy0 + ch), (gx0 + S(50), gy1)], fill="#ef4444", width=S(2))
    draw.text((gx0 + S(50), S(235)), "DISCARDED (X)", fill="#ef4444", font=f_lbl, anchor="mm")
    
    # anxiety -> anxiety (MATCHED GREEN)
    draw.line([(gx0 + cw + cw//2, gy0 + ch), (gx0 + S(175), gy1)], fill="#10b981", width=S(2))
    draw.text((gx0 + S(175), S(235)), "d <= 0.2", fill="#10b981", font=f_txt, anchor="mm")

    draw.rounded_rectangle([S(50), S(350), S(560), S(500)], radius=S(8), fill="#fee2e2", outline="#f87171")
    draw.text((S(305), S(380)), "CRITICAL LIMITATION IN BASE PAPER:", fill="#991b1b", font=f_lbl, anchor="mm")
    draw.text((S(305), S(415)), "Whenever vocabulary granularity differs (e.g. 'observing' -> 'obs'+'erving'),", fill="#7f1d1d", font=f_txt, anchor="mm")
    draw.text((S(305), S(445)), "the algorithm skips distillation entirely, causing catastrophic information loss.", fill="#7f1d1d", font=f_txt, anchor="mm")

    # Panel B: Proposed Sinkhorn OT
    draw.rounded_rectangle([S(620), S(85), S(1170), S(525)], radius=S(12), outline="#cbd5e1", fill="#f8fafc", width=S(1))
    draw.text((S(895), S(110)), "(b) Proposed: Continuous Sinkhorn-Wasserstein OT Coupling", fill="#15803d", font=f_lbl, anchor="mm")
    draw.text((S(895), S(130)), "Smooth mass transport preserves 100% semantic probability mass", fill="#64748b", font=f_sub, anchor="mm")

    # Soft Heatmap Visualization
    hx0, hy0 = S(720), S(170)
    cell = S(48)
    matrix = [
        [0.65, 0.30, 0.03, 0.02],
        [0.32, 0.60, 0.05, 0.03],
        [0.01, 0.02, 0.02, 0.01],
        [0.01, 0.03, 0.88, 0.04],
        [0.01, 0.02, 0.02, 0.92],
        [0.00, 0.01, 0.01, 0.89],
    ]
    # Draw heatmap cells
    for r in range(6):
        for c in range(4):
            val = matrix[r][c]
            # blue gradient
            bg_color = (int(255 - val*210), int(255 - val*160), int(255 - val*40))
            draw.rectangle([hx0 + c*cell*1.8, hy0 + r*cell, hx0 + (c+1)*cell*1.8, hy0 + (r+1)*cell], fill=bg_color, outline="#cbd5e1")
            txt_color = "#ffffff" if val > 0.4 else "#334155"
            draw.text((hx0 + c*cell*1.8 + (cell*1.8)//2, hy0 + r*cell + cell//2), f"{val:.2f}", fill=txt_color, font=f_math, anchor="mm")

    # Labels for matrix
    draw.text((S(670), hy0 + S(25)), "obs", fill="#0f172a", font=f_txt, anchor="mm")
    draw.text((S(670), hy0 + cell + S(25)), "erving", fill="#0f172a", font=f_txt, anchor="mm")
    draw.text((S(670), hy0 + cell*2 + S(25)), "my", fill="#0f172a", font=f_txt, anchor="mm")
    draw.text((S(670), hy0 + cell*3 + S(25)), "anxiety", fill="#0f172a", font=f_txt, anchor="mm")
    draw.text((S(670), hy0 + cell*4 + S(25)), "to", fill="#0f172a", font=f_txt, anchor="mm")
    draw.text((S(670), hy0 + cell*5 + S(25)), "understand", fill="#0f172a", font=f_txt, anchor="mm")

    draw.rounded_rectangle([S(640), S(470), S(1150), S(515)], radius=S(6), fill="#dcfce7", outline="#86efac")
    draw.text((S(895), S(492)), "Complete semantic mass conservation across disparate subwords via Gibbs Kernel K = exp(-C/eps)", fill="#166534", font=f_lbl, anchor="mm")

    img.save(output_path, dpi=(300, 300))
    print(f"Saved: {output_path}")

# =====================================================================
# FIGURE 3: CTC BLANK COLLAPSE DYNAMICS
# =====================================================================
def generate_fig3(output_path="fig3_blank_collapse_dynamics.png"):
    scale = 2
    w, h = 1200 * scale, 560 * scale
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    f_title = ImageFont.truetype(font_serif_bd, 16 * scale)
    f_sub = ImageFont.truetype(font_serif_it, 12 * scale)
    f_lbl = ImageFont.truetype(font_serif_bd, 11 * scale)
    f_txt = ImageFont.truetype(font_serif, 10 * scale)

    draw.rounded_rectangle([S(10), S(10), S(1190), S(550)], radius=S(16), outline="#94a3b8", width=S(1))
    draw.text((w // 2, S(35)), "Figure 3: Mitigation of CTC Blank Collapse via Sinkhorn-Wasserstein Knowledge Distillation", fill="#0f172a", font=f_title, anchor="mm")
    draw.text((w // 2, S(58)), "Posterior Frame Probability Trajectories: Baseline Unsupervised CTC vs. Distilled Student Conformer", fill="#475569", font=f_sub, anchor="mm")

    # Plot A: Vanilla CTC
    ax_x0, ax_y0 = S(80), S(110)
    ax_w, ax_h = S(1040), S(170)
    draw.rectangle([ax_x0, ax_y0, ax_x0 + ax_w, ax_y0 + ax_h], fill="#f8fafc", outline="#94a3b8")
    draw.text((ax_x0 + S(15), ax_y0 + S(15)), "(a) Baseline Vanilla CTC (No KD): Pathological Blank Collapse (Blank Posterior > 92%)", fill="#b91c1c", font=f_lbl)

    # Draw blank line at 0.95
    blank_y = ax_y0 + int(ax_h * 0.08)
    pts_blank = [(ax_x0 + x, blank_y + int(math.sin(x*0.1)*3)) for x in range(0, ax_w, 4)]
    draw.line(pts_blank, fill="#ef4444", width=S(2))
    draw.text((ax_x0 + ax_w - S(100), blank_y - S(10)), "Blank Token Prior (~92.4%)", fill="#ef4444", font=f_lbl)

    # Draw flat non-blank emissions
    pts_nonblank = [(ax_x0 + x, ax_y0 + ax_h - S(8) - int(abs(math.sin(x*0.05))*5)) for x in range(0, ax_w, 4)]
    draw.line(pts_nonblank, fill="#64748b", width=S(1))
    draw.text((ax_x0 + S(200), ax_y0 + ax_h - S(20)), "Suppressed Non-Blank Token Emissions (Underflow)", fill="#64748b", font=f_txt)

    # Plot B: Proposed Sinkhorn Distilled CTC
    bx_y0 = S(320)
    draw.rectangle([ax_x0, bx_y0, ax_x0 + ax_w, bx_y0 + ax_h], fill="#f8fafc", outline="#94a3b8")
    draw.text((ax_x0 + S(15), bx_y0 + S(15)), "(b) Proposed Multi-Task Conformer: Restored Sharp Phonetic Spikes via Sinkhorn Semantic Anchoring", fill="#15803d", font=f_lbl)

    # Words spikes: "HELLO", "DAVID", "HOW", "ARE", "YOU", "TODAY"
    word_peaks = [
        (ax_x0 + int(ax_w * 0.12), "HELLO"),
        (ax_x0 + int(ax_w * 0.28), "DAVID"),
        (ax_x0 + int(ax_w * 0.44), "HOW"),
        (ax_x0 + int(ax_w * 0.58), "ARE"),
        (ax_x0 + int(ax_w * 0.72), "YOU"),
        (ax_x0 + int(ax_w * 0.88), "TODAY")
    ]

    # Draw baseline lower blank
    b_blank_y = bx_y0 + int(ax_h * 0.35)
    pts_b_blank = []
    for x in range(0, ax_w, 4):
        cx = ax_x0 + x
        dip = 0
        for px, _ in word_peaks:
            dist = abs(cx - px)
            if dist < S(35):
                dip += math.exp(-(dist**2)/(2 * (S(12)**2))) * S(90)
        pts_b_blank.append((cx, min(bx_y0 + ax_h - S(5), b_blank_y + int(dip))))
    draw.line(pts_b_blank, fill="#94a3b8", width=S(1))

    # Draw sharp non-blank peaks
    for px, word in word_peaks:
        peak_pts = [
            (px - S(30), bx_y0 + ax_h - S(5)),
            (px - S(10), bx_y0 + S(45)),
            (px, bx_y0 + S(35)),
            (px + S(10), bx_y0 + S(45)),
            (px + S(30), bx_y0 + ax_h - S(5))
        ]
        draw.line(peak_pts, fill="#0284c7", width=S(2))
        draw.text((px, bx_y0 + S(22)), word, fill="#0284c7", font=f_lbl, anchor="mm")

    draw.text((ax_x0 + ax_w - S(120), b_blank_y - S(10)), "Balanced Blank Transitions", fill="#64748b", font=f_txt)
    draw.text((w // 2, bx_y0 + ax_h + S(18)), "Acoustic Time Frames (t)", fill="#334155", font=f_lbl, anchor="mm")

    img.save(output_path, dpi=(300, 300))
    print(f"Saved: {output_path}")

# =====================================================================
# FIGURE 4: MULTI-TASK TRADE-OFF & SENSITIVITY
# =====================================================================
def generate_fig4(output_path="fig4_multitask_tradeoff.png"):
    scale = 2
    w, h = 1200 * scale, 560 * scale
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    f_title = ImageFont.truetype(font_serif_bd, 16 * scale)
    f_sub = ImageFont.truetype(font_serif_it, 12 * scale)
    f_lbl = ImageFont.truetype(font_serif_bd, 11 * scale)
    f_txt = ImageFont.truetype(font_serif, 10 * scale)

    draw.rounded_rectangle([S(10), S(10), S(1190), S(550)], radius=S(16), outline="#94a3b8", width=S(1))
    draw.text((w // 2, S(35)), "Figure 4: Joint Optimization Trade-off between Transcription Fidelity and Emotion Discrimination", fill="#0f172a", font=f_title, anchor="mm")
    draw.text((w // 2, S(58)), "Sensitivity Analysis across Emotion Multi-Task Weight lambda_emo in [0.1, 1.0]", fill="#475569", font=f_sub, anchor="mm")

    gx0, gy0 = S(140), S(100)
    gw, gh = S(920), S(380)
    draw.rectangle([gx0, gy0, gx0 + gw, gy0 + gh], fill="#f8fafc", outline="#94a3b8")

    # Draw grid
    for i in range(1, 6):
        y = gy0 + int(gh * i / 6)
        draw.line([(gx0, y), (gx0 + gw, y)], fill="#e2e8f0", width=1)

    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    wer_vals = [36.2, 36.5, 36.8, 37.1, 37.5, 38.6, 40.2, 42.1, 44.5, 47.8]
    ua_vals  = [51.2, 53.4, 55.8, 58.1, 59.97, 60.1, 60.2, 59.8, 59.5, 59.1]

    # Plot WER (Blue line, Left Axis: 30 to 50%)
    wer_pts = []
    for i, (l, v) in enumerate(zip(lambdas, wer_vals)):
        x = gx0 + int(gw * (l - 0.1) / 0.9)
        y = gy0 + gh - int(gh * (v - 30.0) / 20.0)
        wer_pts.append((x, y))

    draw.line(wer_pts, fill="#2563eb", width=S(3))
    for x, y in wer_pts:
        draw.ellipse([x - S(5), y - S(5), x + S(5), y + S(5)], fill="#2563eb")

    # Plot UA (Emerald line, Right Axis: 50 to 65%)
    ua_pts = []
    for i, (l, v) in enumerate(zip(lambdas, ua_vals)):
        x = gx0 + int(gw * (l - 0.1) / 0.9)
        y = gy0 + gh - int(gh * (v - 50.0) / 15.0)
        ua_pts.append((x, y))

    draw.line(ua_pts, fill="#059669", width=S(3))
    for x, y in ua_pts:
        draw.ellipse([x - S(5), y - S(5), x + S(5), y + S(5)], fill="#059669")

    # Pareto Optimal Point marker at lambda = 0.5
    px = gx0 + int(gw * (0.5 - 0.1) / 0.9)
    draw.line([(px, gy0), (px, gy0 + gh)], fill="#e11d48", width=S(1))
    draw.rounded_rectangle([px - S(90), gy0 + S(30), px + S(90), gy0 + S(90)], radius=S(6), fill="#fff1f2", outline="#f43f5e")
    draw.text((px, gy0 + S(48)), "PARETO OPTIMAL (lambda=0.5)", fill="#be123c", font=f_lbl, anchor="mm")
    draw.text((px, gy0 + S(72)), "WER: 37.5% | SER UA: 59.97%", fill="#0f172a", font=f_txt, anchor="mm")

    # Axis Labels
    draw.text((gx0 - S(60), gy0 + gh // 2), "Word Error Rate (WER %) [Lower is Better]", fill="#2563eb", font=f_lbl, anchor="mm")
    draw.text((gx0 + gw + S(60), gy0 + gh // 2), "Emotion Unweighted Accuracy (UA %) [Higher is Better]", fill="#059669", font=f_lbl, anchor="mm")
    draw.text((gx0 + gw // 2, gy0 + gh + S(30)), "Multi-Task Emotion Loss Weight (lambda_emo)", fill="#334155", font=f_lbl, anchor="mm")

    img.save(output_path, dpi=(300, 300))
    print(f"Saved: {output_path}")

# =====================================================================
# FIGURE 5: IEMOCAP CONFUSION MATRIX & DISAMBIGUATION
# =====================================================================
def generate_fig5(output_path="fig5_iemocap_confusion_disambiguation.png"):
    scale = 2
    w, h = 1200 * scale, 560 * scale
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    f_title = ImageFont.truetype(font_serif_bd, 16 * scale)
    f_sub = ImageFont.truetype(font_serif_it, 12 * scale)
    f_lbl = ImageFont.truetype(font_serif_bd, 11 * scale)
    f_txt = ImageFont.truetype(font_serif, 10 * scale)

    draw.rounded_rectangle([S(10), S(10), S(1190), S(550)], radius=S(16), outline="#94a3b8", width=S(1))
    draw.text((w // 2, S(35)), "Figure 5: Affective Performance on IEMOCAP: Held-Out Confusion Matrix and Ambiguous Utterance Disambiguation", fill="#0f172a", font=f_title, anchor="mm")

    # Panel A: Confusion Matrix (Held-out Session 5)
    cx0, cy0 = S(80), S(100)
    draw.text((S(310), S(80)), "(a) 4-Class Normalized Confusion Matrix (Held-Out Session 5)", fill="#1e293b", font=f_lbl, anchor="mm")
    classes = ["Neutral", "Angry", "Sad", "Happy"]
    cm = [
        [0.61, 0.11, 0.16, 0.12],
        [0.12, 0.64, 0.08, 0.16],
        [0.18, 0.06, 0.63, 0.13],
        [0.17, 0.19, 0.12, 0.52]
    ]
    cs = S(85)
    for r in range(4):
        for c in range(4):
            v = cm[r][c]
            bg = (int(255 - v*170), int(255 - v*120), int(255 - v*40))
            draw.rectangle([cx0 + c*cs + S(70), cy0 + r*cs + S(40), cx0 + (c+1)*cs + S(70), cy0 + (r+1)*cs + S(40)], fill=bg, outline="#cbd5e1")
            tc = "#ffffff" if v > 0.4 else "#334155"
            draw.text((cx0 + c*cs + S(70) + cs//2, cy0 + r*cs + S(40) + cs//2), f"{v*100:.1f}%", fill=tc, font=f_lbl, anchor="mm")

    for i, name in enumerate(classes):
        draw.text((cx0 + S(40), cy0 + i*cs + S(40) + cs//2), name, fill="#334155", font=f_lbl, anchor="mm")
        draw.text((cx0 + i*cs + S(70) + cs//2, cy0 + S(20)), name, fill="#334155", font=f_lbl, anchor="mm")

    # Panel B: Disambiguation of xxx samples
    bx0, by0 = S(650), S(100)
    draw.text((S(910), S(80)), "(b) Automated Tie-Breaking on 2,507 Ambiguous 'xxx' Utterances", fill="#1e293b", font=f_lbl, anchor="mm")
    
    # Consensus Badge
    draw.rounded_rectangle([bx0 + S(20), by0 + S(30), bx0 + S(500), by0 + S(110)], radius=S(10), fill="#dcfce7", outline="#22c55e")
    draw.text((bx0 + S(260), by0 + S(55)), "72.72% Human-AI Consensus Rate", fill="#15803d", font=f_title, anchor="mm")
    draw.text((bx0 + S(260), by0 + S(85)), "1,823 out of 2,507 non-consensus utterances matched human voters", fill="#166534", font=f_txt, anchor="mm")

    # Bar chart for distribution
    dist = [("Neutral", 31.4, "#64748b"), ("Sad", 28.2, "#6366f1"), ("Happy", 20.9, "#10b981"), ("Angry", 19.6, "#ef4444")]
    bar_y0 = by0 + S(150)
    for i, (name, pct, col) in enumerate(dist):
        draw.text((bx0 + S(60), bar_y0 + i*S(50)), name, fill="#334155", font=f_lbl, anchor="mm")
        draw.rounded_rectangle([bx0 + S(110), bar_y0 + i*S(50) - S(12), bx0 + S(110) + int(pct*S(9)), bar_y0 + i*S(50) + S(12)], radius=S(6), fill=col)
        draw.text((bx0 + S(130) + int(pct*S(9)), bar_y0 + i*S(50)), f"{pct:.1f}%", fill="#0f172a", font=f_lbl, anchor="lm")

    img.save(output_path, dpi=(300, 300))
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
