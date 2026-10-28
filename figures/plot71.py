"""R71 figures: (1) error-propagation flows, (2) alignment heatmap, (3) t-SNE state map, (4) per-epoch state
composition, (5) retrieval strips. Usage: plot71.py <out_dir> [strip_qid]"""
import json, sys, os, textwrap, collections
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, FancyBboxPatch, Rectangle
out = Path(sys.argv[1]); BASE = os.environ['R70_BASE']
TF, OP, GOLD, WRONG, INK, MUTE = '#d0645c', '#3a66c9', '#3f8f5e', '#c4524a', '#24304a', '#5b6680'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#9aa3b5', 'axes.linewidth': 0.8,
                     'xtick.color': MUTE, 'ytick.color': MUTE, 'axes.labelcolor': INK, 'pdf.fonttype': 42})
O = dict(np.load(out / 'own70d.npz')); Hd = np.load(BASE + '/round29/inputs/design/doc_emb.npy').astype(np.float32)
titles = [r['title'] for r in json.load(open(BASE + '/round29/inputs/design/corpus.json'))]
qtext = {r['id']: r['question'] for r in json.load(open(BASE + '/round29/inputs/design/raw.json'))}
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

# ---------- (2) alignment heatmap on each encoder's own chain states ----------
COLS = ['latest\nwrong', 'latest\ngold', 'target\n$a^*$', 'later\ngold']
M = {}
for n in ENC:
    acc = collections.defaultdict(list)
    for i in range(N):
        g = G[i]; hist = []; w = [int(d) for d in O[f'{n}_w'][i]]
        for k in range(4):
            miss = [d for d in g if d not in hist]
            if not miss: break
            q = O[f'{n}_q'][i][k].astype(np.float32); nw = min(sum(d not in g for d in hist), 2)
            wr = [d for d in hist if d not in g]; gh = [d for d in hist if d in g]
            vals = [q @ Hd[wr[-1]] if wr else np.nan, q @ Hd[gh[-1]] if gh else np.nan, q @ Hd[miss[0]], q @ Hd[miss[1]] if len(miss) > 1 else np.nan]
            if k > 0: acc[nw].append(vals)
            hist.append(w[k])
    M[n] = {r: (np.nanmean(np.array(v, dtype=float), 0), len(v)) for r, v in acc.items()}
vmin = min(np.nanmin(M[n][r][0]) for n in ENC for r in M[n]); vmax = max(np.nanmax(M[n][r][0]) for n in ENC for r in M[n])
fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.6), gridspec_kw=dict(wspace=0.55))
for ax, n in zip(axes, ENC):
    A = np.array([M[n][r][0] for r in (0, 1, 2)]); im = ax.imshow(A, cmap='Blues', vmin=vmin, vmax=vmax, aspect='auto')
    for r in range(3):
        for c in range(4):
            if np.isnan(A[r, c]): ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc='#f3f4f7', ec='white', hatch='///', lw=0)); continue
            ax.text(c, r, f'{A[r, c]:.2f}', ha='center', va='center', fontsize=8, color='white' if A[r, c] > (vmin + vmax) / 2 else INK, fontweight='bold' if c == 2 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(COLS, fontsize=7); ax.set_yticks(range(3))
    ax.set_yticklabels([f'clean (n={M[n][0][1]})', f'1 wrong (n={M[n][1][1]})', f'2+ wrong (n={M[n][2][1]})'], fontsize=7.4)
    ax.set_title(NAME[n], fontsize=9, color=INK); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.set_label('mean cos(f(s), h(·))', fontsize=7.5); cb.outline.set_visible(False)
fig.savefig(out / 'fig_align.pdf', bbox_inches='tight'); plt.close(fig)
stats['align'] = {n: {str(r): [None if np.isnan(x) else round(float(x), 3) for x in M[n][r][0]] + [M[n][r][1]] for r in M[n]} for n in ENC}

# ---------- (3) t-SNE of all own-chain states (gold still missing), both encoders jointly ----------
if os.environ.get('R71_SKIP_TSNE'): raise SystemExit(json.dump(stats, open(out / 'stats71_partial.json', 'w'), indent=1))
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
pts, meta = [], []
for n in ENC:
    for i in range(N):
        g = G[i]; hist = []; w = [int(d) for d in O[f'{n}_w'][i]]
        for k in range(4):
            if all(d in hist for d in g): break
            pts.append(O[f'{n}_q'][i][k].astype(np.float32)); meta.append((n, i, k, min(sum(d not in g for d in hist), 2))); hist.append(w[k])
X = np.stack(pts); meta = np.array(meta, dtype=object)
for variant in ('raw', 'delta'):
    Z = X.copy()
    if variant == 'delta':
        q0 = {(n, i): O[f'{n}_q'][i][0].astype(np.float32) for n in ENC for i in range(N)}
        Z = np.stack([X[j] - q0[(meta[j][0], meta[j][1])] for j in range(len(X))]); keep = np.array([m[2] > 0 for m in meta]); Z, mm = Z[keep], meta[keep]
    else: mm = meta
    Y = TSNE(n_components=2, metric='cosine', perplexity=40, init='pca', random_state=13).fit_transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True, sharey=True)
    for ax, n in zip(axes, ENC):
        me = np.array([m[0] == n for m in mm]); ax.scatter(Y[~me, 0], Y[~me, 1], s=2, color='#e3e6ec', lw=0)
        for nw, col, lab in ((0, GOLD, 'clean history'), (1, '#e59a3a', '1 wrong passage'), (2, WRONG, '2+ wrong passages')):
            sel_ = me & np.array([m[3] == nw for m in mm]); ax.scatter(Y[sel_, 0], Y[sel_, 1], s=4, color=col, alpha=0.75, lw=0, label=f'{lab} ({sel_.sum()})')
        ax.set_title(f'{NAME[n]}: its own test-time states', fontsize=9, color=INK); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color('#dfe3ea')
        ax.legend(loc='lower left', fontsize=6.8, frameon=False, markerscale=3)
    fig.tight_layout(); fig.savefig(out / f'fig_tsne_{variant}.pdf', bbox_inches='tight'); plt.close(fig)
    # segregation: share of contaminated states' 10 nearest neighbours (same encoder, cosine, full space) that are clean
    seg = {}
    for n in ENC:
        me = np.array([m[0] == n for m in mm]); Zn = Z[me]; con = np.array([m[3] > 0 for m in mm[me]])
        if con.all() or not con.any(): continue
        nn = NearestNeighbors(n_neighbors=11, metric='cosine').fit(Zn); _, I = nn.kneighbors(Zn[con])
        seg[n] = round(100 * float((~con[I[:, 1:]]).mean()), 1)
    stats[f'tsne_{variant}_clean_neighbours_of_contaminated'] = seg

# ---------- (4) per-epoch composition (ours, seed 17) ----------
E = json.load(open(out / 'epochs70.json'))
fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9), sharey=True)
for ax, key, ttl in ((axes[0], 'train', 'states it trains on (training questions)'), (axes[1], 'val', 'states it visits on held-out questions')):
    rows = []
    for e in E:
        c = (e['train'].get('OP-refresh') or e['train']['RN-on-policy']) if key == 'train' else e['val']; c = {int(k): v for k, v in c.items()}; t = sum(c.values())
        rows.append([100 * c.get(k, 0) / t for k in (0, 1, 2)])
    R_ = np.array(rows); x = np.arange(1, len(E) + 1); b = np.zeros(len(E))
    for k, col, lab in ((0, GOLD, 'clean'), (1, '#e59a3a', '1 wrong'), (2, WRONG, '2+ wrong')):
        ax.bar(x, R_[:, k], bottom=b, color=col, width=0.72, edgecolor='white', label=lab)
        for xi, (bb, v) in enumerate(zip(b, R_[:, k])):
            if v > 6: ax.text(x[xi], bb + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=6.6, color='white')
        b += R_[:, k]
    ax.set_xticks(x); ax.set_xlabel('epoch'); ax.set_title(ttl, fontsize=8.5, color=INK); ax.set_ylim(0, 100)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    stats[f'epochs_{key}_contaminated'] = [round(100 - r[0], 1) for r in R_]
axes[0].set_ylabel('share of states (%)'); axes[1].legend(loc='upper center', bbox_to_anchor=(-0.05, 1.28), ncol=3, frameon=False, fontsize=7.4)
fig.tight_layout(); fig.savefig(out / 'fig_epochs.pdf', bbox_inches='tight'); plt.close(fig)
stats['epochs_val_hit_after_mistake'] = [e['val_hit'].get('1') for e in E]; stats['epochs_val_C5'] = [e['val_C5'] for e in E]

# ---------- (5) retrieval strips ----------
qid = sys.argv[2] if len(sys.argv) > 2 else None
i = list(O['qids']).index(qid) if qid else max((i for i in range(N) if len(G[i]) == 4), key=lambda i: sum(int(d) in G[i] for d in O['OP_w'][i]) - sum(int(d) in G[i] for d in O['TF_w'][i]))
g = G[i]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 2.9))
for ax, n in zip(axes, ENC):
    hist = []
    for k in range(4):
        miss = [d for d in g if d not in hist]; y = 3 - k
        ax.text(-0.08, y + 0.4, f't={k + 1}', ha='right', va='center', fontsize=7.6, color=MUTE)
        for c, d in enumerate(O[f'{n}_top5'][i][k]):
            d = int(d); tgt = bool(miss) and d == miss[0]
            fc = GOLD if tgt else ('#cfe8d8' if d in miss else ('#eef0f4' if d in g else '#f6dcd9'))
            ec = INK if c == 0 else 'white'
            ax.add_patch(FancyBboxPatch((c * 1.0 + 0.04, y + 0.08), 0.92, 0.64, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec=ec, lw=1.4 if c == 0 else 0.8))
            ax.text(c + 0.5, y + 0.4, textwrap.shorten(titles[d], 19, placeholder='…'), ha='center', va='center', fontsize=6.2, color='white' if tgt else INK)
        hist.append(int(O[f'{n}_w'][i][k]))
    ax.set_xlim(-0.35, 5.05); ax.set_ylim(-0.1, 4.0); ax.axis('off')
    ax.set_title(f'{NAME[n]}: top 5 at each step (framed: retrieved)', fontsize=8.5, color=INK)
fig.suptitle(textwrap.fill(qtext[str(O['qids'][i])], 130) + '\nGold chain: ' + ' → '.join(titles[d] for d in g), fontsize=7.6, color=INK, y=1.06)
h = [Rectangle((0, 0), 1, 1, fc=GOLD, label='target a* (next missing gold)'), Rectangle((0, 0), 1, 1, fc='#cfe8d8', label='later missing gold'),
     Rectangle((0, 0), 1, 1, fc='#f6dcd9', label='non-gold')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / 'fig_strip.pdf', bbox_inches='tight'); plt.close(fig)
stats['strip_qid'] = str(O['qids'][i])
json.dump(stats, open(out / 'stats71.json', 'w'), indent=1); print(json.dumps(stats, indent=1))
