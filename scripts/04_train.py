"""Entrenamiento, selección de hiperparámetros y diagnóstico (convergencia, sobreajuste, curva de aprendizaje).

Reglas del protocolo:
  - Entrenamiento y validación se separan por instancia (nunca por decisión) para evitar fuga de información.
  - Los hiperparámetros se eligen por el gap medio de makespan en validación, no por accuracy.
  - El conjunto de test (benchmarks) no se toca aquí.
"""
import json, pickle, time
import numpy as np
import joblib
from _common import ROOT
from jsspc.instances import Instance
from jsspc.models import LRModel, RFModel, LGBMModel, Ensemble, make_chooser, make_gated_chooser, top1_accuracy
from jsspc.simulator import CLASSIC
from jsspc.evaluate import eval_rule
from jsspc.features import FEATURE_NAMES

(ROOT / "models").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
tr, va = np.load(ROOT / "data/decisions_train.npz"), np.load(ROOT / "data/decisions_val.npz")
Xtr, ytr, gtr, itr, sztr = (tr[k] for k in ("X", "y", "g", "inst", "size"))
Xva, yva, gva, sizeva = (va[k] for k in ("X", "y", "g", "size"))
refs_val = pickle.load(open(ROOT / "data/ref/ref_val.pkl", "rb"))
val_insts = [Instance(r["name"], r["machines"], r["times"], bks=r["ref_makespan"]) for r in refs_val]
out = {"n_train_rows": int(len(ytr)), "n_val_rows": int(len(yva))}


def val_gap(model):
    df = eval_rule(val_insts, lambda i: make_chooser(model), "x")
    return float(df.gap_pct.mean()), df


# --- 1) Reglas clásicas en validación (para elegir la regla de reserva del híbrido) ---------------
classic_val = {}
for r, f in CLASSIC.items():
    df = eval_rule(val_insts, lambda i, f=f: f, r)
    classic_val[r] = float(df.gap_pct.mean())
best_classic = min((k for k in classic_val if k != "FIFO_est"), key=classic_val.get)
out["classic_val_gap"] = classic_val; out["best_classic_val"] = best_classic
print("clásicas en validación:", {k: round(v, 1) for k, v in classic_val.items()}, "->", best_classic, flush=True)

# --- 2) Rejilla de hiperparámetros ----------------------------------------------------------------
grids = {
    "LR": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
    "RF": [dict(max_depth=d, min_samples_leaf=l) for d, l in ((6, 20), (10, 10), (14, 5), (20, 2))],
    "LGBM": [dict(num_leaves=nl, learning_rate=0.05, min_child_samples=mc) for nl, mc in ((7, 40), (15, 40), (31, 40), (63, 20))],
}
cls = {"LR": LRModel, "RF": RFModel, "LGBM": LGBMModel}
best = {}
out["grid"] = {}
for fam, grid in grids.items():
    out["grid"][fam] = []
    for hp in grid:
        t0 = time.perf_counter()
        mdl = cls[fam](**hp)
        if fam == "LGBM":
            mdl.fit(Xtr, ytr, Xva, yva)
        else:
            mdl.fit(Xtr, ytr)
        tt = time.perf_counter() - t0
        row = dict(hp=hp, train_seconds=tt, train_acc=top1_accuracy(mdl, Xtr, ytr, gtr),
                   val_acc=top1_accuracy(mdl, Xva, yva, gva))
        if fam == "LGBM":
            row["best_iteration"] = int(mdl.best_iteration)
        row["val_gap_pct"], _ = val_gap(mdl)
        out["grid"][fam].append(row)
        print(fam, row, flush=True)
        if fam not in best or row["val_gap_pct"] < best[fam][0]:
            best[fam] = (row["val_gap_pct"], mdl, hp, row)
    joblib.dump(best[fam][1], ROOT / f"models/{fam}.joblib")
    out[f"best_{fam}"] = {"hp": best[fam][2], **{k: v for k, v in best[fam][3].items() if k != "hp"}}

# --- 3) Convergencia de LightGBM (curvas por iteración) -------------------------------------------
gb = best["LGBM"][1]
out["lgbm_curve"] = {k: v["binary_logloss"] for k, v in gb.evals.items()}
out["lgbm_best_iteration"] = int(gb.best_iteration)

# --- 4) Curva de aprendizaje: nº de instancias de entrenamiento ----------------------------------
rng = np.random.default_rng(0)
ids = np.unique(itr); rng.shuffle(ids)
lc = []
for frac in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
    sel = np.isin(itr, ids[: max(2, int(frac * len(ids)))])
    for fam, mk in (("LR", lambda: LRModel(**best["LR"][2])), ("RF", lambda: RFModel(**best["RF"][2])),
                    ("LGBM", lambda: LGBMModel(**best["LGBM"][2]))):
        m = mk()
        if fam == "LGBM":
            m.fit(Xtr[sel], ytr[sel], Xva, yva)
        else:
            m.fit(Xtr[sel], ytr[sel])
        row = dict(model=fam, frac=frac, n_instances=int(len(ids) * frac) if frac < 1 else len(ids),
                   rows=int(sel.sum()), train_acc=top1_accuracy(m, Xtr[sel], ytr[sel], gtr[sel]),
                   val_acc=top1_accuracy(m, Xva, yva, gva))
        if fam == "LGBM":
            row["val_gap_pct"], _ = val_gap(m)
        lc.append(row); print(row, flush=True)
out["learning_curve"] = lc

# --- 5) Exactitud por tamaño en validación (¿generaliza entre tamaños?) ---------------------------
out["val_acc_by_size"] = {}
for fam in ("LR", "RF", "LGBM"):
    d = {}
    for s in np.unique(sizeva):
        sel = sizeva == s
        # reindexar grupos de la selección
        d[f"{s//100}x{s%100}"] = top1_accuracy(best[fam][1], Xva[sel], yva[sel], gva[sel])
    out["val_acc_by_size"][fam] = d

# --- 6) Importancia de características ------------------------------------------------------------
out["importance"] = {
    "feature": FEATURE_NAMES,
    "LGBM_gain": (gb.b.feature_importance("gain", iteration=gb.best_iteration) / gb.b.feature_importance("gain", iteration=gb.best_iteration).sum()).tolist(),
    "RF": best["RF"][1].m.feature_importances_.tolist(),
    "LR_coef": best["LR"][1].m.steps[1][1].coef_[0].tolist(),
}

# --- 7) Conjunto RF+LGBM y umbral de confianza del híbrido ---------------------------------------
ens = Ensemble(best["RF"][1], best["LGBM"][1])
joblib.dump(ens, ROOT / "models/ENS.joblib")
out["ens_val_gap_pct"], _ = val_gap(ens)
out["ens_val_acc"] = top1_accuracy(ens, Xva, yva, gva)
gate = []
for tau in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
    chs = []
    def fac(i):
        c = make_gated_chooser(ens, best_classic, tau); chs.append(c); return c
    df = eval_rule(val_insts, fac, "gate")
    ml = sum(c.stats["ml"] for c in chs); fb = sum(c.stats["fallback"] for c in chs)
    gate.append(dict(tau=tau, val_gap_pct=float(df.gap_pct.mean()), fallback_share=fb / max(1, ml + fb)))
    print(gate[-1], flush=True)
out["gate"] = gate
out["tau_best"] = min(gate, key=lambda r: r["val_gap_pct"])["tau"]

# Portafolio (ML y mejor clásica; se queda con el mejor plan) en validación
dml = eval_rule(val_insts, lambda i: make_chooser(ens), "ml")
dcl = eval_rule(val_insts, lambda i: CLASSIC[best_classic], "cl")
out["portfolio_val_gap_pct"] = float((np.minimum(dml.makespan, dcl.makespan) / dml.bks * 100 - 100).mean())
(ROOT / "results/training.json").write_text(json.dumps(out, indent=1, default=float))
print("hecho; tau* =", out["tau_best"], "portfolio val gap", out["portfolio_val_gap_pct"])
