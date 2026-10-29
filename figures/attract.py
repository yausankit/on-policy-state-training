"""Figure 7: cosine of contaminated queries to the wrong passage in their history and to the target. Input: viz70.json
from viz_attract.py. Usage: attract.py <out_dir>"""
import json, sys, textwrap
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde

out = Path(sys.argv[1]); V = json.load(open(out / 'viz70.json'));
TF, OP, GOLD, WRONG, INK = '#d0645c', '#3a66c9', '#3f8f5e', '#b8433b', '#24304a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#9aa3b5', 'axes.linewidth': 0.8,
                     'xtick.color': '#5b6680', 'ytick.color': '#5b6680', 'axes.labelcolor': INK, 'pdf.fonttype': 42})
def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
stats = {}


# ---------- (2) mistake attraction ----------
A = V['attraction']
x = {n: np.array([a[n]['wrong'] for a in A]) for n in ('TF', 'OP')}; y = {n: np.array([a[n]['target'] for a in A]) for n in ('TF', 'OP')}
fig = plt.figure(figsize=(7.4, 3.5)); gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 4.2], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=ax); axr = fig.add_subplot(gs[1, 1], sharey=ax)
lo, hi = min(x['TF'].min(), x['OP'].min(), y['TF'].min(), y['OP'].min()) - .02, max(x['TF'].max(), x['OP'].max(), y['TF'].max(), y['OP'].max()) + .02
gx, gy = np.mgrid[lo:hi:120j, lo:hi:120j]
for n, col in (('TF', TF), ('OP', OP)):
    ax.scatter(x[n], y[n], s=3, color=col, alpha=0.18, linewidth=0)
    k = gaussian_kde(np.vstack([x[n], y[n]])); z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contour(gx, gy, z, levels=np.quantile(z[z > z.max() * 0.02], [0.5, 0.8, 0.95]), colors=col, linewidths=[0.7, 1.0, 1.4])
    ax.scatter(np.median(x[n]), np.median(y[n]), marker='D', s=40, color=col, edgecolor='white', zorder=5)
    for data, a_, orient in ((x[n], axt, 'v'), (y[n], axr, 'h')):
        g = np.linspace(lo, hi, 200); d = gaussian_kde(data)(g)
        (a_.fill_between(g, d, color=col, alpha=0.25, lw=0) if orient == 'v' else a_.fill_betweenx(g, d, color=col, alpha=0.25, lw=0))
        (a_.plot(g, d, color=col, lw=1) if orient == 'v' else a_.plot(d, g, color=col, lw=1))
    stats[f'attr_above_diag_{n}'] = round(100 * float(np.mean(y[n] > x[n])), 1)
    stats[f'attr_median_{n}'] = dict(wrong=round(float(np.median(x[n])), 3), target=round(float(np.median(y[n])), 3))
ax.plot([lo, hi], [lo, hi], color='#9aa3b5', lw=0.8, ls=':'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.text(hi - .01, lo + .01, 'closer to its own mistake', ha='right', va='bottom', fontsize=7.5, color='#5b6680')
ax.text(lo + .01, hi - .01, 'closer to the target', ha='left', va='top', fontsize=7.5, color='#5b6680')
ax.set_xlabel(r'cos$(f(s),\,h(\mathrm{wrong\ passage\ in\ history}))$'); ax.set_ylabel(r'cos$(f(s),\,h(a^*))$')
for a_ in (axt, axr): a_.axis('off')
clean(ax)
# right panel: displacement field of the same states, binned on the teacher-forced position
ax2 = fig.add_subplot(gs[:, 2]); nb = 5; ex = np.linspace(np.quantile(x['TF'], .01), np.quantile(x['TF'], .99), nb + 1); ey = np.linspace(np.quantile(y['TF'], .01), np.quantile(y['TF'], .99), nb + 1)
dx, dy = x['OP'] - x['TF'], y['OP'] - y['TF']; X_, Y_, U_, W_, C_, N_ = [], [], [], [], [], []
for a_ in range(nb):
    for b_ in range(nb):
        m = (x['TF'] >= ex[a_]) & (x['TF'] < ex[a_ + 1]) & (y['TF'] >= ey[b_]) & (y['TF'] < ey[b_ + 1])
        if m.sum() >= 20:
            X_.append((ex[a_] + ex[a_ + 1]) / 2); Y_.append((ey[b_] + ey[b_ + 1]) / 2); U_.append(dx[m].mean()); W_.append(dy[m].mean()); C_.append(dy[m].mean() - dx[m].mean()); N_.append(m.sum())
lim = max(abs(np.array(C_)).max(), 1e-3)

q_ = ax2.quiver(X_, Y_, U_, W_, color=OP, angles='xy', scale_units='xy', scale=1, width=0.012, headwidth=3.4, headlength=3.8, zorder=3)
ax2.scatter(X_, Y_, s=14, color=TF, zorder=4, linewidth=0)
ax2.plot([lo, hi], [lo, hi], color='#9aa3b5', lw=0.8, ls=':'); ax2.set_xlim(ex[0] - .03, ex[-1] + .03); ax2.set_ylim(ey[0] - .03, ey[-1] + .03)
ax2.set_xlabel('cos to wrong passage (teacher-forced)'); ax2.yaxis.set_label_position('right'); ax2.yaxis.tick_right(); ax2.set_ylabel(r'cos to target $a^*$ (teacher-forced)')
ax2.set_title('mean shift of the same states: teacher-forced (dot) → ours (arrow head)', fontsize=8, color=INK)
for sp in ('top', 'left'): ax2.spines[sp].set_visible(False)
h = [plt.Line2D([], [], color=TF, marker='o', ls='', label='teacher-forced encoder'), plt.Line2D([], [], color=OP, marker='o', ls='', label='ours')]
fig.legend(handles=h, loc='upper center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
fig.savefig(out / 'fig_attract.pdf', bbox_inches='tight'); plt.close(fig)
stats['attr_states'] = len(A)
stats['attr_shift_up_left'] = round(100 * float(np.mean((y['OP'] - x['OP']) > (y['TF'] - x['TF']))), 1)
print(json.dumps(stats, indent=1))
