import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_enhanced_diagram():
    # High-resolution figure: 18 x 7.5 inches at 300 DPI = 5400 x 2250 pixels!
    fig, ax = plt.subplots(figsize=(17, 7.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette matching user's original diagram
    c_bg_box = '#ffffff'
    c_blue_head = '#1d4ed8'
    c_blue_box = '#eff6ff'
    c_blue_border = '#3b82f6'
    
    c_purple_head = '#7e22ce'
    c_purple_box = '#faf5ff'
    c_purple_border = '#a855f7'
    
    c_amber_head = '#b45309'
    c_amber_box = '#fffbeb'
    c_amber_border = '#f59e0b'
    
    c_green_head = '#047857'
    c_green_box = '#ecfdf5'
    c_green_border = '#10b981'
    
    c_red_head = '#b91c1c'
    c_red_box = '#fef2f2'
    c_red_border = '#ef4444'
    
    c_gray_head = '#334155'
    c_gray_box = '#f8fafc'
    c_gray_border = '#64748b'

    # Main Diagram Title
    ax.text(50, 96.5, "Dynamic Knowledge Distillation Transfer for Multimodal Speech Recognition and Emotion Classification",
            fontsize=13, fontweight='bold', ha='center', va='center', color='#0f172a')

    # Helper function to draw rounded boxes
    def draw_card(x, y, w, h, title, subtitle_list, border_col, fill_col, title_bg=None, title_col='#000000'):
        # Main background box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                      facecolor=fill_col, edgecolor=border_col, linewidth=1.5)
        ax.add_patch(rect)
        
        # If title bar
        if title_bg:
            thead = patches.FancyBboxPatch((x, y + h - 4.5), w, 4.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                                           facecolor=title_bg, edgecolor=border_col, linewidth=1.2)
            ax.add_patch(thead)
            ax.text(x + w/2, y + h - 2.3, title, fontsize=9.5, fontweight='bold', ha='center', va='center', color='#ffffff')
            curr_y = y + h - 7.0
        else:
            ax.text(x + w/2, y + h - 2.8, title, fontsize=9.5, fontweight='bold', ha='center', va='center', color=title_col)
            curr_y = y + h - 6.5
            
        for line in subtitle_list:
            weight = 'bold' if line.startswith('[') or line.startswith('1)') or line.startswith('2)') or line.startswith('L =') else 'normal'
            fsize = 8.2 if len(line) < 30 else 7.5
            ax.text(x + w/2, curr_y, line, fontsize=fsize, fontweight=weight, ha='center', va='center', color='#1e293b')
            curr_y -= 3.3

    # ================= 1. LEFT: DATASET & INPUTS =================
    # Dataset Master Box
    draw_card(2, 63, 17, 26, "Dataset", ["IEMOCAP & LibriSpeech"], c_gray_border, c_gray_box, title_bg=c_gray_head)
    # Audio Wave Sub-box
    draw_card(3.5, 71.5, 14, 9.5, "Audio Wave", ["16 kHz 16-bit WAV Audio"], '#2563eb', '#ffffff', title_col='#1d4ed8')
    # Input Transcript Sub-box
    draw_card(3.5, 50, 14, 9.5, "Input Transcript", ["Reference Text"], '#7e22ce', '#ffffff', title_col='#7e22ce')

    # ================= 2. STUDENT ASR MODEL =================
    draw_card(23, 62, 18, 28, "STUDENT ASR MODEL", [
        "Conformer Backbone",
        "(6 Layers, Decoder-Less)",
        "[4x Subsampling, K=15]",
        "Frame Latent Matrix",
        "Z in R^{T' x 256}"
    ], c_blue_border, c_blue_box, title_bg=c_blue_head)

    # ================= 3. FROZEN TEACHER =================
    draw_card(23, 22, 18, 28, "FROZEN TEACHER", [
        "LLaMA-3.2-1B Model (Frozen)",
        "[128k Vocab, 2048 Dim]",
        "Pseudo-Bidirectional Approach",
        "Context Window L' = 8",
        "Grad theta_T = 0 (Static Anchor)"
    ], c_purple_border, c_purple_box, title_bg=c_purple_head)

    # ================= 4. CORE DISTILLATION NOVELTY (SINKHORN OT) =================
    draw_card(45, 48, 19, 42, "CORE DISTILLATION NOVELTY", [
        "Optimal Transport Plan P*",
        "Many-to-Many Token Alignment",
        "1) Ground Cost Matrix C",
        "2) Top-K Semantic Filtering",
        "3) Entropic Regularization (eps=0.10)",
        "Sinkhorn-Knopp Scaling",
        "Wasserstein OT Loss:",
        "L_OT = <P*, C> - eps H(P)"
    ], c_amber_border, c_amber_box, title_bg=c_amber_head)

    # ================= 5. DUAL OUTPUT HEADS (ASR & SER) =================
    # ASR Prediction
    draw_card(68, 77, 14, 13, "ASR Decoding Head", [
        "Predicted Tokens (Y_hat)",
        "Linear Projection (Vocab 1024)",
        "CTC Loss Optimization"
    ], c_blue_border, c_blue_box, title_col=c_blue_head)

    # Emotion Classifier
    draw_card(68, 56, 14, 14, "Emotion Classifier", [
        "Temporal Pooling + MLP",
        "(4-Class Softmax)",
        "Neutral, Anger, Sad, Happy"
    ], c_green_border, c_green_box, title_col=c_green_head)

    # ================= 6. MULTI-TASK OBJECTIVE =================
    draw_card(45, 14, 19, 23, "MULTI-TASK OBJECTIVE", [
        "Total Joint Loss L",
        "L = L_CTC + 0.1*L_OT + 0.2*L_Emo",
        "Backprop to Student Only",
        "theta_S <- theta_S - eta grad L"
    ], c_red_border, c_red_box, title_bg=c_red_head)

    # ================= 7. EVALUATION METRICS & PREDICTIONS =================
    draw_card(85, 60, 13.5, 30, "EVALUATION METRICS", [
        "ASR Metrics:",
        "- CER (%) = 23.4%",
        "- WER (%) = 37.5%",
        "Emotion Metrics:",
        "- WA (%) = 57.70%",
        "- UA (%) = 59.97%",
        "- xxx Consensus: 72.72%"
    ], c_gray_border, c_gray_box, title_bg=c_gray_head)

    draw_card(85, 20, 13.5, 28, "SYSTEM PREDICTIONS", [
        "System Output:",
        "1) Recognized Speech",
        "   Transcribed Text",
        "2) Emotion Label",
        "   (4 Affective States)",
        "Real-Time Edge:",
        "RTF_CPU = 0.08"
    ], c_blue_border, c_blue_box, title_bg='#1e40af')

    # ================= ARROWS & FLOWS =================
    def draw_arrow(x1, y1, x2, y2, color='#334155', style='->', ls='-', label=None, label_pos=(0,0)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.6, linestyle=ls, shrinkA=3, shrinkB=3))
        if label:
            lx, ly = (x1 + x2)/2 + label_pos[0], (y1 + y2)/2 + label_pos[1]
            ax.text(lx, ly, label, fontsize=7.5, fontweight='bold', ha='center', va='center', color=color,
                    bbox=dict(boxstyle='square,pad=0.2', facecolor='#ffffff', edgecolor='none'))

    # Audio Wave -> Conformer
    draw_arrow(17.5, 76, 23, 76, color='#2563eb', label='Mel-Filterbanks')
    
    # Input Transcript -> Teacher
    draw_arrow(17.5, 55, 23, 38, color='#7e22ce', label='Token IDs')

    # Conformer -> OT Plan
    draw_arrow(41, 76, 45, 76, color='#2563eb', label='Frame Latents')

    # Teacher -> OT Plan
    draw_arrow(41, 38, 48, 48, color='#7e22ce', label='Causal Prior')

    # Conformer -> ASR Head
    draw_arrow(32, 90, 68, 83.5, color='#2563eb')

    # Conformer -> Emotion Classifier
    draw_arrow(41, 68, 68, 63, color='#059669', label='Acoustic Rep.')

    # ASR Head -> Loss
    draw_arrow(75, 77, 64, 25, color='#b91c1c', label='L_CTC', label_pos=(-2, 0))

    # Emotion Head -> Loss
    draw_arrow(75, 56, 64, 28, color='#b91c1c', label='L_Emo', label_pos=(1, 0))

    # OT Plan -> Loss
    draw_arrow(54.5, 48, 54.5, 37, color='#b91c1c', label='L_OT')

    # Loss backprop -> Student Conformer (Dashed arrow)
    ax.annotate('', xy=(32, 62), xytext=(45, 25),
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.8, linestyle='--', shrinkA=2, shrinkB=2))
    ax.text(37, 43, 'Backprop: Update\nStudent Weights Only', fontsize=7.2, fontweight='bold', ha='center', va='center',
            color='#dc2626', bbox=dict(boxstyle='round,pad=0.2', facecolor='#fee2e2', edgecolor='#ef4444', lw=0.8))

    # ASR Head -> Evaluation
    draw_arrow(82, 83.5, 85, 80, color='#334155')

    # Emotion Head -> Evaluation
    draw_arrow(82, 63, 85, 68, color='#334155')

    # Heads to Predictions
    draw_arrow(82, 80, 85, 38, color='#1d4ed8')
    draw_arrow(82, 60, 85, 30, color='#047857')

    plt.tight_layout()
    out_path = r"C:\Users\Deepak R G\.gemini\antigravity\scratch\FINAL-YEAR-PROJECT\system_diagram_enhanced.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Saved enhanced diagram to:", out_path)

if __name__ == '__main__':
    draw_enhanced_diagram()
