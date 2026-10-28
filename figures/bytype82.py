"""Per-type retrieval heatmap (appendix). Values from round71/tables.md (ours: mean over seeds 42-46).
MuSiQue: R@5 / C@5 by hops; long-chain: R@10 / C@10 by type."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
M = ['BM25', 'Dense', 'G-Retriever$^\\dagger$', 'HippoRAG 2$^\\dagger$', 'LinearRAG', 'MDR', 'Q-RAG', 'Search-R1', 'Ours']
COLS = ['2-hop', '3-hop', '4-hop', '3-hop', '4-hop', '5-hop', '3-hop', '4-hop', '5-hop', 'comp.']
N = ['518', '316', '166', '100', '100', '100', '170', '45', '12', '73']
GROUPS = [('MuSiQue (@5)', 0, 3), ('HotpotQA-Long (@10)', 3, 6), ('2Wiki-Long (@10)', 6, 10)]
R = np.array([
 [51.0, 39.8, 26.5, 34.0, 25.8, 16.8, 31.2, 31.1, 23.3, 33.8],
 [63.3, 53.5, 32.5, 42.3, 29.2, 24.0, 37.8, 37.2, 26.7, 26.3],
 [46.6, 30.1, 17.2, 27.3, 20.2, 14.0, 27.3, 26.7, 21.7, 7.9],
 [67.5, 55.3, 35.1, 44.7, 30.2, 25.2, 39.6, 37.2, 26.7, 27.3],
 [53.5, 36.6, 27.7, 29.0, 22.0, 15.2, 30.0, 27.2, 21.7, 30.1],
 [71.4, 51.5, 30.3, 72.7, 54.2, 41.6, 69.4, 36.7, 36.7, 21.5],
 [71.0, 58.9, 38.0, 39.7, 27.0, 17.4, 31.8, 25.6, 28.3, 19.7],
 [80.4, 58.1, 40.2, 75.3, 59.2, 40.8, 82.9, 43.3, 40.0, 53.3],
 [81.0, 75.2, 46.7, 77.5, 54.7, 37.6, 77.3, 48.2, 44.7, 54.5]])
C = np.array([
 [17.6, 1.9, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [32.2, 7.0, 0.0, 3.0, 0.0, 0.0, 1.2, 6.7, 8.3, 0.0],
 [17.6, 0.3, 0.0, 5.0, 1.0, 0.0, 2.9, 0.0, 0.0, 0.0],
 [39.8, 7.9, 0.0, 4.0, 0.0, 0.0, 1.8, 6.7, 8.3, 0.0],
 [30.5, 6.0, 0.0, 2.0, 0.0, 0.0, 1.2, 2.2, 0.0, 0.0],
 [54.6, 11.1, 0.6, 35.0, 8.0, 1.0, 24.7, 2.2, 0.0, 0.0],
 [55.0, 20.9, 6.6, 12.0, 4.0, 1.0, 4.1, 0.0, 0.0, 0.0],
 [65.3, 8.5, 0.0, 49.0, 23.0, 5.0, 65.3, 0.0, 0.0, 0.0],
 [67.7, 41.6, 6.1, 56.8, 24.6, 7.4, 55.6, 5.8, 3.3, 1.4]])
INK, OURS = '#24304a', '#3a66c9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.05), gridspec_kw=dict(wspace=0.04))
for ax, A, name, cmap in ((axes[0], R, 'Recall (R@k)', 'Blues'), (axes[1], C, 'Complete coverage (C@k)', 'Purples')):
    Z = A / np.maximum(A.max(0, keepdims=True), 1e-9)
    ax.imshow(Z, cmap=cmap, vmin=0, vmax=1.15, aspect='auto')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            best = A[i, j] == A[:, j].max() and A[:, j].max() > 0
            ax.text(j, i, f'{A[i, j]:.1f}', ha='center', va='center', fontsize=6.6, color='white' if Z[i, j] > 0.62 else INK, fontweight='bold' if best else None)
            if best: ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.2))
    ax.add_patch(Rectangle((-.5, len(M) - 1.5), A.shape[1], 1, fill=False, ec=OURS, lw=2.0, clip_on=False))
    for x in (2.5, 5.5): ax.axvline(x, color='white', lw=3)
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([f'{c}\n({n})' for c, n in zip(COLS, N)], fontsize=6.6)
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M if ax is axes[0] else [''] * len(M))
    for lab in ax.get_yticklabels():
        if lab.get_text() == 'Ours': lab.set_color(OURS); lab.set_fontweight('bold')
    ax.tick_params(length=0)
    for g, a, b in GROUPS: ax.text((a + b - 1) / 2, -0.95, g, ha='center', va='bottom', fontsize=7.2, color=INK, fontweight='bold')
    ax.set_title(name, fontsize=8.5, color=INK, pad=24, fontweight='bold')
    for sp in ax.spines.values(): sp.set_visible(False)
fig.savefig('/data1/xuboyan/rev/round82/bytype_purple.pdf', bbox_inches='tight', pad_inches=0.02)
