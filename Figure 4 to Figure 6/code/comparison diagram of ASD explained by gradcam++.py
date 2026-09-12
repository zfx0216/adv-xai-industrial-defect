import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

eps = [2, 4, 8]

# ===================== Simulated ASD data (LayerCAM, matching the table values)=====================
# Data structure: data[attack][train_type][model]
# attack:0=FGSM, 1=PGD, 2=MI-FGSM
# train_type:0=Vanilla,1=PGD-AT
# model:0=ResNet50,1=DenseNet121,2=ViT-B/16
# ---------------- Untargeted (U) ----------------
data_untarget = [
    # FGSM
    [
        [[14.9873, 15.0813, 17.4042], [13.5826, 13.6704, 16.6036], [35.3274, 43.1454, 45.2798]],  # Vanilla
        [[14.7527, 14.9905, 17.0283], [13.1832, 13.3448, 16.2056], [35.2626, 29.6165, 35.7085]]   # PGD-AT
    ],
    # PGD
    [
        [[29.2117, 31.2546, 31.2884], [35.6197, 41.0739, 41.0984], [40.1326, 43.1754, 42.7711]],
        [[26.0755, 27.9749, 28.1432], [30.4157, 32.2752, 32.4341], [39.8696, 37.1708, 34.0103]]
    ],
    # MI-FGSM
    [
        [[28.7716, 31.4911, 31.5067], [27.1408, 28.2898, 28.1254], [38.0602, 42.1470, 47.4529]],
        [[25.4559, 28.0786, 28.4066], [27.0575, 27.6549, 27.9730], [37.8973, 34.7332, 37.6072]]
    ]
]
# ---------------- Targeted (T) ----------------
data_target = [
    # FGSM
    [
        [[13.9231, 13.8998, 16.7303], [12.8695, 12.8748, 16.2472], [34.9822, 39.0023, 42.6782]],
        [[13.4628, 13.4635, 12.0395], [12.7262, 12.7436, 15.5064], [34.1249, 29.6637, 34.9603]]
    ],
    # PGD
    [
        [[25.2617, 28.6549, 28.9496], [40.3539, 37.2377, 37.7474], [40.3917, 43.6629, 45.5615]],
        [[25.1116, 28.6139, 28.9311], [27.9098, 30.5557, 31.0303], [39.9747, 41.0478, 35.2244]]
    ],
    # MI-FGSM
    [
        [[25.0284, 29.1275, 29.4843], [28.3536, 30.3914, 30.4929], [33.2760, 41.4376, 48.5762]],
        [[23.3956, 28.7200, 29.0021], [26.7685, 29.8278, 30.3564], [33.2463, 36.6355, 41.0150]]
    ]
]

# Style configuration
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
markers = ["o", "s", "^"]
model_names = ["ResNet50", "DenseNet121", "ViT-B/16"]
attack_names = ["FGSM", "PGD", "MI-FGSM"]

# Create a canvas with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)

# Plot in a loop
for row_idx, data_row in enumerate([data_untarget, data_target]):
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]
        attack_data = data_row[col_idx]
        van_data = attack_data[0]
        at_data = attack_data[1]
        for m_idx in range(3):
            ax.plot(eps, van_data[m_idx], linestyle="-", marker=markers[m_idx], color=colors[m_idx], lw=1.2, label=f"{model_names[m_idx]} Vanilla")
            ax.plot(eps, at_data[m_idx], linestyle="--", marker=markers[m_idx], color=colors[m_idx], lw=1.2, label=f"{model_names[m_idx]} PGD-AT")
        ax.set_title(f"{attack_names[col_idx]}", fontsize=11)
        ax.set_xticks(eps)
        ax.set_xlabel(r"$\epsilon$ (/255)")
        # Add the Y-axis label only to the first column to avoid repetition
        if col_idx == 0:
            ax.set_ylabel("Attention Semantic Drift (ASD)")
        ax.grid(alpha=0.3)

# Row titles
axes[0, 1].annotate("Untargeted Attack", xy=(0.5, 1.12), xycoords="axes fraction", ha="center", fontsize=12)
axes[1, 1].annotate("Targeted Attack", xy=(0.5, 1.12), xycoords="axes fraction", ha="center", fontsize=12)

# Unified legend (placed outside the plot)
handles, labels = axes[0, 0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc="outside left upper", fontsize=9)

plt.savefig(r"F:\IndustrialInspectionCode\ASD_comparison_charts\asd_all_attack_ut_gradcam.pdf", bbox_inches="tight")
plt.show()