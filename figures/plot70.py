"""R70 plots from viz70.json / examples70.json. Usage: plot70.py <out_dir>"""
import json, sys, textwrap
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde

out = Path(sys.argv[1]); V = json.load(open(out / 'viz70.json')); E = json.load(open(out / 'examples70.json'))
TF, OP, GOLD, WRONG, INK = '#d0645c', '#3a66c9', '#3f8f5e', '#b8433b', '#24304a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#9aa3b5', 'axes.linewidth': 0.8,
                     'xtick.color': '#5b6680', 'ytick.color': '#5b6680', 'axes.labelcolor': INK, 'pdf.fonttype': 42})
def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
stats = {}

# ---------- (1) trajectories (plane of the four gold passages) ----------
import os
BASE = os.environ.get('R70_BASE', '/home/amax/revision_20260928')
Hd = np.load(BASE + '/round29/inputs/design/doc_emb.npy', mmap_mode='r'); titles = [r['title'] for r in json.load(open(BASE + '/round29/inputs/design/corpus.json'))]
O = np.load(out / 'own70.npz'); qtext = {t['qid']: t['question'] for t in V['trajectories']}
rows = []
for i, q in enumerate(O['qids']):
    g = list(O['chain'][i]); found = {n: sum(int(d) in g for d in O[f'{n}_w'][i]) for n in ('TF', 'OP')}
    G = np.asarray(Hd[g], dtype=np.float32); c = G.mean(0); U = np.linalg.svd(G - c, full_matrices=False)[2][:2]
    Q = np.concatenate([O['TF_q'][i], O['OP_q'][i]]) - c; fid = float(((Q @ U.T) ** 2).sum() / (Q ** 2).sum())
    P2 = {n: (O[f'{n}_q'][i] - c) @ U.T for n in ('TF', 'OP')}; span = float(sum(np.linalg.norm(np.diff(P2[n], axis=0), axis=1).sum() for n in P2))
    rows.append(dict(i=i, q=str(q), g=g, found=found, U=U, c=c, fid=fid, span=span))
cand = [r for r in rows if r['found']['OP'] - r['found']['TF'] >= 2 and r['found']['OP'] >= 3]
cand.sort(key=lambda r: -r['span']); pick = []
for r in cand:
    if all(not set(r['g']) & set(p['g']) for p in pick): pick.append(r)
    if len(pick) == 3: break
stats['traj_candidates'] = len(cand); stats['traj_picked'] = [dict(qid=r['q'], found=r['found'], plane_fidelity=round(r['fid'], 2)) for r in pick]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
for ax, r in zip(axes, pick):
    pr = lambda v: (np.asarray(v, dtype=np.float32) - r['c']) @ r['U'].T; i = r['i']
    wr = {}
    for n in ('TF', 'OP'):
        for d in O[f'{n}_w'][i]:
            if int(d) not in r['g']: wr.setdefault(int(d), set()).add(n)
    for d, by in wr.items():
        col = TF if by == {'TF'} else (OP if by == {'OP'} else INK)
        ax.scatter(*pr(Hd[d]), marker='X', s=46, color=col, alpha=0.75, edgecolor='white', linewidth=0.6, zorder=3)
    for h, d in enumerate(r['g'], 1):
        xy = pr(Hd[d]); ax.scatter(*xy, marker='s', s=120, color=GOLD, edgecolor='white', linewidth=1, zorder=4)
        ax.text(*xy, f'g{h}', ha='center', va='center', fontsize=6.5, color='white', fontweight='bold', zorder=5)
        ax.annotate(textwrap.shorten(titles[d], 24, placeholder='…'), xy, xytext=(0, -11), textcoords='offset points', ha='center', fontsize=6.6, color=GOLD, zorder=6)
    for enc, col, ls in (('TF', TF, (0, (3, 2))), ('OP', OP, '-')):
        pts = [pr(v) for v in O[f'{enc}_q'][i]]
        for k in range(4):
            w = int(O[f'{enc}_w'][i][k])
            ax.plot(*zip(pts[k], pr(Hd[w])), color=col, lw=0.6, alpha=0.35, ls=':', zorder=2)
        for a_, b_ in zip(pts, pts[1:]):
            ax.add_patch(FancyArrowPatch(a_, b_, arrowstyle='-|>', mutation_scale=10, color=col, lw=1.7, linestyle=ls, zorder=4, shrinkA=5, shrinkB=5))
        for k, xy in enumerate(pts, 1):
            ax.scatter(*xy, s=105, color='white', edgecolor=col, linewidth=1.6, zorder=5)
            ax.text(*xy, str(k), ha='center', va='center', fontsize=6.8, color=col, fontweight='bold', zorder=6)
    ax.set_title(textwrap.fill(qtext[r['q']], 54), fontsize=7.4, color=INK, loc='left')
    ax.text(0.98, 0.03, f"gold found: TF {r['found']['TF']}/4, ours {r['found']['OP']}/4", transform=ax.transAxes, ha='right', fontsize=6.8, color='#5b6680')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal', adjustable='datalim'); ax.margins(0.14)
    for sp in ax.spines.values(): sp.set_color('#dfe3ea')
h = [plt.Line2D([], [], color=TF, ls=(0, (3, 2)), lw=1.6, marker='o', mfc='white', label='teacher-forced encoder: query at step t'),
     plt.Line2D([], [], color=OP, lw=1.6, marker='o', mfc='white', label='ours: query at step t'),
     plt.Line2D([], [], color=GOLD, marker='s', ls='', label='gold passage (hop)'), plt.Line2D([], [], color='#8a94aa', marker='X', ls='', label='retrieved non-gold passage'),
     plt.Line2D([], [], color='#8a94aa', ls=':', lw=0.8, label='query → retrieved passage')]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out / 'fig_traj.pdf', bbox_inches='tight'); plt.close(fig)

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

# ---------- (3) attribution: aggregate ----------
agg = {}
for n in ('TF', 'OP'):
    M = V['attribution'][n]; tot = {k: sum(m['mass'][k] for m in M) for k in ('question', 'gold', 'wrong')}; s_ = sum(tot.values())
    tok = {k: sum(m['ntok'][k] for m in M) for k in tot}
    agg[n] = dict(share={k: 100 * v / s_ for k, v in tot.items()}, density={k: (tot[k] / max(tok[k], 1)) / (s_ / sum(tok.values())) for k in tot})
    per = np.array([m['mass']['wrong'] / max(sum(m['mass'].values()), 1e-9) for m in M]); agg[n]['wrong_share_median'] = 100 * float(np.median(per))
stats['attribution'] = {n: {kk: {k: round(v, 2) for k, v in vv.items()} if isinstance(vv, dict) else round(vv, 1) for kk, vv in agg[n].items()} for n in agg}
fig, ax = plt.subplots(figsize=(5.2, 1.5)); cols = {'question': '#8a94aa', 'gold': GOLD, 'wrong': WRONG}
names = {'question': 'question', 'gold': 'gold passages in history', 'wrong': 'wrong passages in history'}
for row, n in enumerate(('OP', 'TF')):
    left = 0
    for k in ('question', 'gold', 'wrong'):
        v = agg[n]['share'][k]; ax.barh(row, v, left=left, color=cols[k], height=0.62, edgecolor='white', label=names[k] if row == 0 else None)
        if v > 6: ax.text(left + v / 2, row, f'{v:.0f}%', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        left += v
ax.set_yticks([0, 1]); ax.set_yticklabels(['ours', 'teacher-forced']); ax.set_xlim(0, 100); ax.set_xlabel('share of |gradient × input| on the state (contaminated states)')
clean(ax); ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3, frameon=False, fontsize=7.5)
fig.savefig(out / 'fig_attr_share.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- (4) representation difference ----------
P = V['projection']; t_ = np.array([p['target_dir'] for p in P]); h_ = np.array([p['hard_dir'] for p in P]); m_ = np.array([p['target_minus_hard'] for p in P])
fig, ax = plt.subplots(figsize=(5.2, 2.3)); g = np.linspace(min(t_.min(), h_.min()) - .02, max(t_.max(), h_.max()) + .02, 300)
for data, col, lab in ((t_, OP, r'onto $h(a^*) - h(\mathrm{wrong})$'), (h_, '#9aa3b5', r'onto $h(\mathrm{hard\ negative}) - h(\mathrm{wrong})$')):
    d = gaussian_kde(data)(g); ax.fill_between(g, d, color=col, alpha=0.28, lw=0); ax.plot(g, d, color=col, lw=1.3, label=lab)
    ax.axvline(np.median(data), color=col, lw=1, ls='--')
ax.axvline(0, color=INK, lw=0.7); ax.set_yticks([]); ax.set_xlabel(r'projection of $f_{\mathrm{ours}}(s) - f_{\mathrm{TF}}(s)$')
clean(ax); ax.spines['left'].set_visible(False); ax.legend(frameon=False, fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.savefig(out / 'fig_delta.pdf', bbox_inches='tight'); plt.close(fig)
stats['delta'] = dict(target_dir_median=round(float(np.median(t_)), 3), target_dir_pos=round(100 * float(np.mean(t_ > 0)), 1),
                      hard_dir_median=round(float(np.median(h_)), 3), hard_dir_pos=round(100 * float(np.mean(h_ > 0)), 1),
                      target_minus_hard_pos=round(100 * float(np.mean(m_ > 0)), 1), norm_median=round(float(np.median([p['norm'] for p in P])), 3))
json.dump(stats, open(out / 'stats70.json', 'w'), indent=1); print(json.dumps(stats, indent=1))
