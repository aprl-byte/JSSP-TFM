"""Convierte las soluciones de referencia en un conjunto de decisiones etiquetadas.

Cada decisión con conflicto (|C| >= 2) genera |C| filas (una por candidata); la candidata que la
referencia programa antes en la máquina lleva y=1. Las decisiones sin conflicto no se incluyen.
"""
import json, pickle
import numpy as np
from _common import ROOT
from jsspc.instances import Instance
from jsspc.labeling import label_instance
from jsspc.features import FEATURE_NAMES

summary = {"features": FEATURE_NAMES}
for split in ("train", "val"):
    refs = pickle.load(open(ROOT / f"data/ref/ref_{split}.pkl", "rb"))
    Xs, ys, gs, iid, size_id = [], [], [], [], []
    off, exact, worse = 0, 0, []
    for k, r in enumerate(refs):
        inst = Instance(r["name"], r["machines"], r["times"])
        X, y, g, mk = label_instance(inst, r["ref_start"])
        exact += mk == r["ref_makespan"]
        worse.append(100 * (mk / r["ref_makespan"] - 1))
        if len(y) == 0:
            continue
        Xs.append(X); ys.append(y); gs.append(g + off); off += g.max() + 1
        iid.append(np.full(len(y), k)); size_id.append(np.full(len(y), r["n"] * 100 + r["m"]))
    X, y, g, iid, sz = map(np.concatenate, (Xs, ys, gs, iid, size_id))
    np.savez_compressed(ROOT / f"data/decisions_{split}.npz", X=X.astype(np.float32), y=y.astype(np.int8),
                        g=g.astype(np.int32), inst=iid.astype(np.int32), size=sz.astype(np.int32))
    summary[split] = {"instances": len(refs), "decisions": int(off), "rows": int(len(y)),
                      "positive_rate": float(y.mean()),
                      "ref_optimal_pct": float(100 * np.mean([r["optimal"] for r in refs])),
                      "replay_exact_pct": float(100 * exact / len(refs)),
                      "replay_mean_excess_pct": float(np.mean(worse)),
                      "replay_max_excess_pct": float(np.max(worse))}
    print(split, summary[split])
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/dataset_summary.json").write_text(json.dumps(summary, indent=1))
