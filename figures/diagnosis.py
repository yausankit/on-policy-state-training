"""Figure 2 (diagnosis): (a) contamination of teacher-forced states along the chain on four-hop MuSiQue questions;
(b) next-step success on the fixed teacher-forced states by the number of wrong passages in the history."""
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
TF, MID, OURS, INK, MUTE = '#c9605a', '#9db8e8', '#3a66c9', '#24304a', '#6b7487'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.edgecolor': '#9aa3b5', 'axes.linewidth': 0.7,
                     'xtick.color': MUTE, 'ytick.color': MUTE, 'axes.labelcolor': INK, 'pdf.fonttype': 42})
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.42))
# (a)
x = np.arange(1, 5); y = np.array([0, 19.3, 69.9, 89.2])
a.fill_between(x, y, color=TF, alpha=0.15, lw=0)
a.plot(x, y, color=TF, lw=1.8, marker='o', ms=4, mfc='white', mew=1.4)
for xi, yi, dx in zip(x, y, (0.0, -0.22, -0.3, 0.0)):
    a.text(xi + dx, yi + 6, f'{yi:.0f}%', ha='center', va='bottom', fontsize=6.6, color=TF, fontweight='bold')
a.set_xticks(x); a.set_xlim(0.6, 4.4); a.set_ylim(0, 112); a.set_yticks([0, 50, 100])
a.set_xlabel('Step'); a.set_ylabel('States with a wrong passage (%)', fontsize=6.4)
a.set_title('(a) Mistakes accumulate', fontsize=7.4, color=INK, loc='left', pad=3)
# (b)
g = np.arange(3); w = 0.26
vals = {'Teacher forcing': ([67.0, 25.0, 14.5], TF), 'w/o refresh': ([67.2, 32.2, 25.1], MID), 'Ours': ([69.0, 35.8, 29.2], OURS)}
for k, (name, (v, c)) in enumerate(vals.items()):
    bars = b.bar(g + (k - 1) * w, v, w, color=c, label=name, zorder=2, edgecolor='white', lw=0.4)
for gi, (t, o) in enumerate(zip(vals['Teacher forcing'][0], vals['Ours'][0])):
    if gi == 0:
        continue
    top = o + 4
    b.plot([gi - w, gi - w, gi + w, gi + w], [t + 1.5, top, top, o + 1.5], color=INK, lw=0.7, zorder=3)
    lab = f'+{o - t:.1f}' if gi == 1 else f'×{o / t:.1f}'
    b.text(gi, top + 1.5, lab, ha='center', va='bottom', fontsize=7, color=OURS, fontweight='bold')
b.set_xticks(g); b.set_xticklabels(['0', '1', '2+']); b.set_ylim(0, 82); b.set_yticks([0, 25, 50, 75])
b.set_xlabel('Wrong passages in history'); b.set_ylabel('Next-step success (%)', fontsize=6.4)
b.set_title('(b) Recovery on the same states', fontsize=7.4, color=INK, loc='left', pad=3)
b.grid(axis='y', color='#eceef3', lw=0.6, zorder=0)
b.legend(frameon=False, fontsize=5.8, loc='upper right', bbox_to_anchor=(1.02, 0.93), handlelength=1, handletextpad=0.4, borderaxespad=0)
for ax in (a, b):
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, width=0.5)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else 'diagnosis.pdf', bbox_inches='tight', pad_inches=0.01)
