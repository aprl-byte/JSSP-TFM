"""Evaluación final en benchmarks no vistos: mismas instancias, mismo simulador, mismos criterios.

Uso: python scripts/05_evaluate.py [--cpsat] [--limit N]
  sin --cpsat : reglas clásicas, aleatoria (5 semillas) y modelos aprendidos en OR-Library, Taillard y DMU
  con  --cpsat: CP-SAT (límite 10 s) en el subconjunto FT/ABZ/ORB/LA y ta01-ta10
"""
import argparse, json, time
import numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
from _common import ROOT
from jsspc.instances import read_instance, load_bks
from jsspc.simulator import CLASSIC, make_random_rule
from jsspc.models import make_chooser, make_gated_chooser
from jsspc.evaluate import eval_rule
from jsspc.reference import solve_cpsat

BKS = load_bks(ROOT / "data/bks.json")
FAM_ORLIB = ["ft", "abz", "orb", "la", "swv", "yn"]


def load_sets(limit=None):
    sets = {"ORLIB": [], "TA": [], "DMU": []}
    for fam in FAM_ORLIB:
        for p in sorted((ROOT / f"data/raw/{fam}").glob("*.jsp")):
            sets["ORLIB"].append(p)
    sets["TA"] = sorted((ROOT / "data/raw/ta").glob("ta*.jsp"))
    sets["DMU"] = sorted((ROOT / "data/raw/dmu").glob("dmu*.jsp"))
    out = {}
    for k, ps in sets.items():
        ps = ps[:limit] if limit else ps
        lst = []
        for p in ps:
            i = read_instance(p, p.stem, family=k)
            i.bks, i.bks_optimal = BKS[p.stem][0], BKS[p.stem][1]
            lst.append(i)
        out[k] = lst
    return out


def run_method(name, insts, tau, fallback):
    """Una tarea = un método sobre todos los conjuntos (para paralelizar por método)."""
    if name in CLASSIC:
        fac = lambda i: CLASSIC[name]
    elif name == "RANDOM":
        dfs = []
        for s in range(5):
            d = eval_rule(insts, lambda i, s=s: make_random_rule(1000 * s + i.n * 7 + i.m), "RANDOM", True)
            d["seed"] = s; dfs.append(d)
        return pd.concat(dfs)
    elif name in ("LR", "RF", "LGBM", "ENS"):
        mdl = joblib.load(ROOT / f"models/{name}.joblib")
        fac = lambda i: make_chooser(mdl)
    elif name == "HYB":
        mdl = joblib.load(ROOT / "models/ENS.joblib")
        fac = lambda i: make_gated_chooser(mdl, fallback, tau)
    return eval_rule(insts, fac, name, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpsat", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.1,
                    help="umbral del híbrido; 0.1 (no el óptimo en validación, que fue 0.0 = sin reserva)")
    a = ap.parse_args()
    sets = load_sets(a.limit)
    insts = [i for s in sets.values() for i in s]
    (ROOT / "results").mkdir(exist_ok=True)
    if a.cpsat:
        sel = [i for i in insts if i.name[:2] in ("ft", "ab", "or", "la") or i.name in [f"ta{k:02d}" for k in range(1, 11)]]
        rows = []
        for i in sel:
            cm, opt, secs, _ = solve_cpsat(i, 10.0, workers=2, seed=0)
            rows.append(dict(method="CPSAT10", instance=i.name, family=i.family, size=i.size, n=i.n, m=i.m,
                             makespan=cm, bks=i.bks, seconds=secs, optimal=opt,
                             gap_pct=100 * (cm / i.bks - 1)))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results/eval_cpsat.csv", index=False)
        return
    tr = json.loads((ROOT / "results/training.json").read_text())
    tau, fb = a.tau, tr["best_classic_val"]
    methods = ["FIFO", "SPT", "LPT", "MWKR", "LWKR", "MOPNR", "RANDOM", "LR", "RF", "LGBM", "ENS", "HYB"]
    t0 = time.time()
    dfs = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_method)(m, insts, tau, fb) for m in methods)
    df = pd.concat(dfs)
    df.to_csv(ROOT / "results/eval_instances.csv", index=False)
    json.dump({"tau": tau, "fallback": fb, "seconds": time.time() - t0}, open(ROOT / "results/eval_meta.json", "w"))
    print(df.groupby(["method"]).gap_pct.mean().round(2))


if __name__ == "__main__":
    main()
