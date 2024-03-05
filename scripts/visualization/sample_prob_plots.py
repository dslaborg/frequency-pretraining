"""
This script is used to recreate parts of Figure 1 of our manuscript.
It creates two bar plots, one for visualizing the probability distribution of the frequency bins and one for the sleep
stages.
"""
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

sns.set_style("white")

# barplot 1: frequency bins

# initialize some test data
sample_freqs = [
    (0.3, 0.6),
    (0.6, 1.2),
    (1.2, 2.3),
    (2.3, 4.6),
    (4.6, 9.0),
    (9.0, 17.7),
    (17.7, 35.0),
]
sample_freq_probabilities = [0.1, 0.2, 0.7, 0.6, 0.4, 0.9, 0.2]

plt.figure(figsize=(3, 2.3))
plt.bar(
    [f"{f[0]}-{f[1]}" for f in sample_freqs],
    sample_freq_probabilities,
    color=["grey"] * 2 + ["green"] * 2 + ["grey", "green", "grey"],
)

# add dashed line at 0.5 as the cutoff threshold
plt.hlines(0.5, -0.6, 6.6, colors="red", linestyles="dashed")
plt.text(1.4, 0.55, "threshold", color="red", ha="right", va="center")

plt.xticks(np.arange(len(sample_freqs)) - 0.2, rotation=30)
plt.xlabel("Frequency Bin (Hz)")
plt.ylabel("Probability")
plt.xlim(-0.6, 6.6)
plt.ylim(0, 1.0)
plt.grid()
plt.tight_layout()

# -----------------

# barplot 2: sleep stages

# initialize some test data
sleepstages = ["W", "N1", "N2", "N3", "REM"]
sleepstage_probabilities = [0.05, 0.18, 0.62, 0.09, 0.06]

plt.figure(figsize=(3, 2))
plt.bar(
    sleepstages, sleepstage_probabilities, color=["grey"] * 2 + ["green"] + ["grey"] * 2
)

# add small red circle around N2 with the text "argmax"
plt.scatter(2, 0.62, color="red", s=20)
plt.text(2, 0.65, "argmax", color="red", ha="center", va="bottom")

plt.xlabel("Sleep Stage")
plt.ylabel("Probability")
plt.ylim(0, 1.0)
plt.grid()
plt.tight_layout()

plt.show()
