# figures

| Script | Figure |
|---|---|
| `bytype.py` | Figure 2, retrieval by question type |
| `diagnosis.py` | Figure 3, contamination along the chain and recovery on fixed states |
| `viz_trajectories.py` → `trajectories.py` | Figure 4, query trajectories |
| `viz_flow.py` → `flow.py` | Figure 6, step-by-step outcomes |
| `viz_attract.py` → `attract.py` | Figure 7, cosine of contaminated queries to the wrong passage and to the target |

`viz_*.py` encode the states of both encoders' own chains (GPU, reads `src/core.py` and `OPST_ROOT/inputs/design`); the plotting scripts read their outputs.
