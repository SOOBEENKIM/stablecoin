"""Plot saved descriptive counts; no fitted predictions appear in this figure."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE/'feasibility_counts.csv')
split = ['initial_training', 'development', 'evaluation']
refs = ['mean5', 'median5', 'BTC']
colors = ['#202020', '#7b7b7b', '#b9b9b9']
names = ['Five-coin mean', 'Five-coin median', 'BTC']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(8, 4.8))
for i, (ref, color, label) in enumerate(zip(refs, colors, names)):
    a = d.loc[d.reference.eq(ref)].set_index('split').reindex(split)
    vals = 100*a.future_lower_rate.to_numpy()
    bars = ax.bar(np.arange(3)+(i-1)*.24, vals, width=.22, color=color, label=label)
    ax.bar_label(bars, labels=[f'{v:.1f}%' for v in vals], padding=3, fontsize=9)
ax.set_xticks(range(3), ['Initial training\n3,389 origins / 145 days',
                       'Development\n1,455 origins / 61 days',
                       'Evaluation\n1,814 origins / 77 days'])
ax.set_ylabel('Observed lower-tail frequency (%)')
ax.set_ylim(0, 34)
ax.set_title('Different reference definitions yield different risk frequencies', pad=14)
ax.legend(frameon=False, loc='upper left', ncol=3, fontsize=9)
ax.grid(axis='y', alpha=.13)
ax.set_axisbelow(True)
fig.text(.105, .035, 'Definition-specific thresholds fixed at pre-November 2025 price quantiles.\n'
         'Outcome: relative basis exactly 6 hours ahead. Descriptive, dependent observations.',
         fontsize=8, color='#444444')
fig.subplots_adjust(left=.105, right=.985, bottom=.23, top=.88)
for ext in ['png', 'pdf']:
    fig.savefig(HERE/f'reference_rates.{ext}', dpi=180)
plt.close(fig)
