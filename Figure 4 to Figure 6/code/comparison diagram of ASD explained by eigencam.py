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
        [[30.8841, 31.0632, 34.3553], [42.2552, 42.9479, 46.9628], [16.8447, 17.2076, 21.5246]],  # Vanilla
        [[25.0160, 27.8926, 32.0232], [35.0844, 36.0904, 43.9905], [16.2305, 17.1607, 20.2615]]   # PGD-AT
    ],
    # PGD
    [
        [[35.6197, 41.0739, 41.0984], [50.9484, 51.9390, 52.2576], [20.7564, 20.6674, 20.6674]],
        [[35.3805, 39.8651, 40.2464], [49.3768, 50.3965, 50.9087], [18.5822, 18.0791, 18.0802]]
    ],
    # MI-FGSM
    [
        [[40.4117, 41.6800, 41.8134], [51.3780, 51.9604, 52.4753], [21.2504, 21.0374, 21.0716]],
        [[40.8550, 41.5540, 41.5711], [49.1919, 50.1832, 50.2309], [18.4064, 18.3981, 18.2862]]
    ]
]
# ---------------- Targeted (T) ----------------
data_target = [
    # FGSM
    [
        [[26.8526, 27.6370, 32.8471], [42.2352, 41.9400, 49.4151], [16.9636, 17.6870, 22.9789]],
        [[24.1639, 24.1647, 24.8909], [38.6621, 38.7664, 43.7476], [16.6362, 17.3052, 20.4819]]
    ],
    # PGD
    [
        [[40.3539, 37.2377, 37.7474], [59.1475, 60.2907, 61.2787], [21.6377, 21.8479, 21.8671]],
        [[36.5306, 37.0982, 37.2336], [54.1440, 56.0157, 56.8422], [20.0202, 20.3025, 20.2713]]
    ],
    # MI-FGSM
    [
        [[36.0278, 37.7071, 38.0126], [59.5686, 58.9595, 59.8896], [20.7417, 20.8180, 21.0635]],
        [[35.4228, 36.3331, 37.9015], [54.2464, 56.2080, 56.7867], [19.7415, 20.4598, 20.4003]]
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

plt.savefig(r"F:\IndustrialInspectionCode\ASD_comparison_charts\asd_all_attack_ut_eigencam.pdf", bbox_inches="tight")
plt.show()