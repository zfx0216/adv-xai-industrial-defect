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
        [[14.9447, 14.9995, 17.1481], [13.3151, 13.4267, 16.2280], [36.2550, 30.5467, 36.9379]],  # Vanilla
        [[14.7652, 14.8700, 16.8263], [13.2781, 13.3892, 15.3673], [32.1495, 29.4072, 36.0251]]   # PGD-AT
    ],
    # PGD
    [
        [[28.2260, 30.1951, 30.2476], [26.4098, 26.9293, 26.8237], [34.5824, 33.7676, 40.1393]],
        [[25.1076, 26.9431, 27.1019], [25.9679, 26.6905, 26.7511], [34.5775, 33.6098, 31.8402]]
    ],
    # MI-FGSM
    [
        [[27.6996, 30.3637, 30.3652], [26.0729, 27.0038, 26.8494], [41.1228, 35.4044, 42.9911]],
        [[24.4885, 26.9778, 27.3046], [25.9868, 26.1431, 26.4226], [36.4046, 31.6003, 34.6928]]
    ]
]
# ---------------- Targeted (T) ----------------
data_target = [
    # FGSM
    [
        [[13.9716, 13.9492, 16.6728], [12.6902, 12.6784, 15.9361], [33.6063, 29.8890, 35.0785]],
        [[13.5865, 13.5869, 12.1912], [11.7844, 11.9030, 15.5928], [31.4083, 27.9675, 34.1757]]
    ],
    # PGD
    [
        [[24.7860, 27.9922, 28.3481], [28.3278, 29.8460, 30.0704], [35.1258, 35.8749, 42.7282]],
        [[23.8993, 27.0643, 27.3813], [27.4157, 28.9185, 29.4193], [34.3327, 35.3024, 36.0376]]
    ],
    # MI-FGSM
    [
        [[24.4848, 28.4291, 28.8007], [28.0032, 29.8194, 29.9140], [38.3327, 35.8828, 44.2345]],
        [[23.4885, 27.1618, 27.4284], [26.2909, 29.1209, 29.6924], [37.0412, 33.8471, 36.3206]]
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

plt.savefig(r"F:\IndustrialInspectionCode\ASD_comparison_charts\asd_all_attack_ut_layercam.pdf", bbox_inches="tight")
plt.show()