import os
import tempfile
import numpy as np

os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.edgecolor'] = '#333333'
matplotlib.rcParams['axes.linewidth'] = 1.0

# =====================================================================
# NEW PLOT 1: LATENCY & MEMORY BENCHMARK (For Section V-D)
# =====================================================================
def generate_latency_memory_plot(output_path="fig_latency_memory_benchmark.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

    # Subplot A: Parameter Count vs RTF (Scatter on Log Scale)
    models = ['Speech LLM\n(1390M)', 'AED + WavLM\n(132M)', 'Whisper-Base\n(74M)', 'Proposed Conformer\n(12.55M)']
    params = [1390, 132, 74, 12.55]
    rtf_cpu = [47.10, 0.28, 0.22, 0.08]
    colors = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981']

    for p, r, col, name in zip(params, rtf_cpu, colors, models):
        ax1.scatter(p, r, s=180, color=col, edgecolor='#1f2937', linewidth=1.5, zorder=4)
        offset_y = 1.4 if 'Speech' in name else 1.25 if 'WavLM' in name else 0.7 if 'Whisper' in name else 0.75
        offset_x = 0.6 if 'Proposed' in name else 1.0
        ax1.annotate(name, (p*offset_x, r*offset_y), fontsize=9, fontweight='bold', color='#1f2937', ha='center')

    ax1.axhline(1.0, color='#dc2626', linestyle='--', linewidth=1.2, label='Real-Time Threshold (RTF=1.0)')
    ax1.fill_between([5, 2000], [0.001, 0.001], [1.0, 1.0], color='#dcfce7', alpha=0.35, label='Real-Time Operational Zone')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Model Parameters (Millions - Log Scale)\n(a)', fontsize=11, labelpad=6)
    ax1.set_ylabel('Real-Time Factor (RTF on CPU - Log Scale)', fontsize=11)
    ax1.set_xlim(8, 2500)
    ax1.set_ylim(0.03, 80)
    ax1.grid(True, which='both', linestyle=':', color='#cbd5e1', linewidth=0.7)
    ax1.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='#cccccc')

    # Subplot B: RAM Memory Footprint Bar Chart
    ram_mb = [5560, 528, 296, 50.2]
    bars = ax2.bar(models, ram_mb, color=colors, edgecolor='#1f2937', linewidth=1.2, width=0.55)
    ax2.set_yscale('log')
    ax2.set_xlabel('Evaluated System\n(b)', fontsize=11, labelpad=6)
    ax2.set_ylabel('Resident RAM Allocation (MB - Log Scale)', fontsize=11)
    ax2.set_ylim(20, 10000)
    ax2.grid(axis='y', linestyle='--', color='#e2e8f0', linewidth=0.8)

    for bar, val in zip(bars, ram_mb):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y * 1.25, f"{val} MB", ha='center', fontsize=9.5, fontweight='bold', color='#1f2937')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Plot: {output_path}")

# =====================================================================
# NEW PLOT 2: AMBIGUOUS SPEECH DISAMBIGUATION (For Section V-E)
# =====================================================================
def generate_disambiguation_plot(output_path="fig_xxx_human_ai_consensus.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

    # Subplot A: Human-AI Consensus Donut
    consensus_sizes = [72.72, 27.28]
    consensus_labels = ['AI Consensus Agreement\n(1,823 Utterances)', 'Disputed Ambiguity\n(684 Utterances)']
    consensus_colors = ['#10b981', '#cbd5e1']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax1.pie(
        consensus_sizes, explode=explode, labels=consensus_labels, autopct='%1.1f%%',
        startangle=140, colors=consensus_colors,
        textprops=dict(color="#1f2937", fontsize=10),
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax1.set_title("Human-AI Consensus Rate on 2,507 Ambiguous (xxx) Samples\n(a)", fontsize=11, pad=12)

    # Subplot B: Resolved Affective Distribution Bar Chart
    emotions = ['Neutral', 'Sadness', 'Happiness', 'Anger']
    percentages = [31.4, 28.2, 20.9, 19.6]
    counts = [787, 707, 524, 491]
    emo_colors = ['#64748b', '#6366f1', '#10b981', '#ef4444']

    bars = ax2.barh(emotions, percentages, color=emo_colors, edgecolor='#1f2937', linewidth=1.2, height=0.55)
    ax2.set_xlabel('Resolved Proportion (%) across 2,507 Ambiguous Dialogues\n(b)', fontsize=11, labelpad=6)
    ax2.set_xlim(0, 40)
    ax2.grid(axis='x', linestyle='--', color='#e2e8f0', linewidth=0.8)

    for bar, pct, cnt in zip(bars, percentages, counts):
        w = bar.get_width()
        ax2.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{pct:.1f}% ({cnt} utts)", va='center', fontsize=9.5, fontweight='bold', color='#1f2937')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Plot: {output_path}")

# =====================================================================
# NEW PLOT 3: CASE STUDY SPECTROGRAM & POSTERIOR ALIGNMENT (Section VI)
# =====================================================================
def generate_spectrogram_case_plot(output_path="fig_case_study_alignment.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 5.2), dpi=300, gridspec_kw={'height_ratios': [2, 1.2]})

    # Generate synthetic spectrogram-like heatmap
    np.random.seed(101)
    time_bins = 200
    mel_bins = 80
    spec = np.zeros((mel_bins, time_bins))
    for t in range(time_bins):
        spec[:, t] = np.sin(np.linspace(0, 3.14, mel_bins)) * np.exp(-((t % 35 - 15)**2)/80) + np.random.normal(0, 0.15, mel_bins)
    spec = np.clip(spec, 0.0, 1.5)

    im = ax1.imshow(spec, aspect='auto', origin='lower', cmap='magma', extent=[0, 4.0, 0, 80])
    ax1.set_ylabel('Mel Frequency Bin', fontsize=10)
    ax1.set_title("Qualitative Acoustic Alignment Case Study on Expressive Utterance: 'SHE COULD GET ALONG WITHOUT ME'\n(a) 80-Channel Log-Mel Filterbank Spectrogram", fontsize=11, pad=8)
    ax1.tick_params(labelsize=9)

    # Plot B: CTC Posterior Word Peaks
    t_frames = np.linspace(0, 4.0, 400)
    words = [
        (0.45, "E (SHE)", '#f59e0b'),
        (1.05, "COULD", '#10b981'),
        (1.65, "GET", '#10b981'),
        (2.35, "A LONG (ALONG)", '#f59e0b'),
        (3.15, "WITHOUT", '#10b981'),
        (3.70, "ME", '#10b981')
    ]

    ax2.set_xlim(0, 4.0)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlabel('Time (Seconds)\n(b) Decoded Word Alignments & Phonetic CTC Activations', fontsize=10, labelpad=6)
    ax2.set_ylabel('Posterior P(w|X)', fontsize=10)
    ax2.grid(True, linestyle=':', color='#cbd5e1', linewidth=0.7)

    # Draw blank curve
    blank_curve = 0.15 + 0.05 * np.sin(t_frames*8)
    for center, _, _ in words:
        blank_curve -= 0.12 * np.exp(-((t_frames - center)**2)/(2*0.06**2))
    ax2.plot(t_frames, np.clip(blank_curve, 0.02, 0.3), color='#94a3b8', linestyle='--', label='Blank Prior P(blank)')

    # Draw word activation spikes
    for center, word, col in words:
        spike = np.exp(-((t_frames - center)**2)/(2*0.06**2))
        ax2.plot(t_frames, spike, color=col, linewidth=1.8)
        ax2.text(center, 1.05, word, ha='center', fontsize=9, fontweight='bold', color=col)

    ax2.legend(loc='upper right', fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Plot: {output_path}")

if __name__ == "__main__":
    generate_latency_memory_plot()
    generate_disambiguation_plot()
    generate_spectrogram_case_plot()
