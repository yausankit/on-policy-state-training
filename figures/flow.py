"""Figure 6: step-by-step outcomes (hit / miss / done) of both encoders on their own chains, 3- and 4-hop MuSiQue
questions. Input: own70d.npz from viz70d.py. Usage: flow.py <out_dir>"""
import json, sys, os, textwrap, collections
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, FancyBboxPatch, Rectangle
out = Path(sys.argv[1]); BASE = os.environ.get('OPST_ROOT', str(Path(__file__).resolve().parents[1] / 'src'))
TF, OP, GOLD, WRONG, INK, MUTE = '#d0645c', '#3a66c9', '#3f8f5e', '#c4524a', '#24304a', '#5b6680'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#9aa3b5', 'axes.linewidth': 0.8,
                     'xtick.color': MUTE, 'ytick.color': MUTE, 'axes.labelcolor': INK, 'pdf.fonttype': 42})
O = dict(np.load(out / 'own70d.npz')); Hd = np.load(BASE + '/inputs/design/doc_emb.npy').astype(np.float32)
titles = [r['title'] for r in json.load(open(BASE + '/inputs/design/corpus.json'))]
qtext = {r['id']: r['question'] for r in json.load(open(BASE + '/inputs/design/raw.json'))}
N = len(O['qids']); G = [[int(d) for d in O['chain'][i] if d >= 0] for i in range(N)]; ENC = ('TF', 'OP'); NAME = {'TF': 'teacher forcing', 'OP': 'ours'}
stats = {}

# ---------- (1) error-propagation flows on 3- and 4-hop questions ----------
LAB = ['hit', 'miss', 'done']; COL = {'hit': GOLD, 'miss': WRONG, 'done': '#b9c0cd'}
def labels(i, n):
    g = G[i]; hist = []; L = []
    for k in range(4):
        w = int(O[f'{n}_w'][i][k]); L.append('done' if all(d in hist for d in g) else ('hit' if w in g else 'miss')); hist.append(w)
    return L
sel = [i for i in range(N) if len(G[i]) >= 3]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
for ax, n in zip(axes, ENC):
    L = [labels(i, n) for i in sel]; tot = len(L); W = 0.24; gap = 0.025 * tot
    ypos = {}
    for k in range(4):
        c = collections.Counter(l[k] for l in L); y = 0
        for lab in LAB:
            ypos[(k, lab)] = [y, y + c[lab]]; y += c[lab] + gap
    for k in range(3):
        fl = collections.Counter((l[k], l[k + 1]) for l in L); outy = {lab: ypos[(k, lab)][0] for lab in LAB}; iny = {lab: ypos[(k + 1, lab)][0] for lab in LAB}
        for a in LAB:
            for b in LAB:
                v = fl[(a, b)]
                if not v: continue
                y0, y1 = outy[a], iny[b]; x0, x1 = k + W / 2, k + 1 - W / 2; xm = (x0 + x1) / 2
                verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + v), (xm, y1 + v), (xm, y0 + v), (x0, y0 + v), (x0, y0)]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
                rec = a == 'miss' and b == 'hit'
                ax.add_patch(PathPatch(MPath(verts, codes), fc=GOLD if rec else COL[a], alpha=0.75 if rec else 0.22, lw=0, zorder=2 if rec else 1))
                outy[a] += v; iny[b] += v
    for (k, lab), (y0, y1) in ypos.items():
        if y1 > y0:
            ax.add_patch(Rectangle((k - W / 2, y0), W, y1 - y0, fc=COL[lab], ec='white', lw=0.8, zorder=3))
            if y1 - y0 > 0.05 * tot: ax.text(k, (y0 + y1) / 2, f'{lab}\n{int(round(y1 - y0))}', ha='center', va='center', fontsize=6.8, color='white', fontweight='bold', zorder=4)
    rec = sum(1 for l in L for k in range(3) if l[k] == 'miss' and l[k + 1] == 'hit'); mis = sum(1 for l in L for k in range(3) if l[k] == 'miss' and l[k + 1] != 'done')
    stats[f'flow_{n}'] = dict(questions=tot, recovery_rate=round(100 * rec / max(mis, 1), 1), misses_followed=mis)
    ax.set_xlim(-0.3, 3.3); ax.set_xticks(range(4)); ax.set_xticklabels([f'step {k + 1}' for k in range(4)]); ax.set_yticks([])
    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
    ax.set_title(f'{NAME[n]}: after a miss, the next step hits in {stats[f"flow_{n}"]["recovery_rate"]:.0f}% of cases', fontsize=8.5, color=INK)
    ax.set_ylim(max(y1 for (y0, y1) in ypos.values()) + gap, -gap)
h = [Rectangle((0, 0), 1, 1, fc=GOLD, label='hit: retrieves a missing gold passage'), Rectangle((0, 0), 1, 1, fc=WRONG, label='miss: retrieves a non-gold passage'),
     Rectangle((0, 0), 1, 1, fc='#b9c0cd', label='done: all gold already retrieved'), Rectangle((0, 0), 1, 1, fc=GOLD, alpha=0.75, label='recovery flow (miss → hit)')]
fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out / 'fig_flow.pdf', bbox_inches='tight'); plt.close(fig)
print(json.dumps(stats, indent=1))
