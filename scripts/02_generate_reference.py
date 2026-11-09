"""Genera instancias sintéticas y sus soluciones de referencia con CP-SAT.

Uso: python scripts/02_generate_reference.py [--time-limit 6] [--jobs 2]
Salida: data/ref/ref_train.pkl y data/ref/ref_val.pkl (listas de diccionarios).
Las semillas de entrenamiento (1000+) y validación (5000+) no se solapan.
"""
import argparse, pickle, time
from joblib import Parallel, delayed
from _common import ROOT
from jsspc.instances import generate
from jsspc.reference import solve_cpsat

# (n, m, nº entrenamiento, nº validación)
PLAN = [(6, 6, 120, 15), (8, 8, 120, 15), (10, 5, 120, 15), (10, 10, 150, 20),
        (12, 10, 110, 15), (15, 5, 100, 10), (15, 10, 130, 20), (20, 5, 50, 10)]


def work(n, m, seed, tl):
    inst = generate(n, m, seed)
    cm, opt, secs, st = solve_cpsat(inst, tl, workers=1, seed=0)
    return dict(name=inst.name, n=n, m=m, seed=seed, machines=inst.machines, times=inst.times,
                ref_start=st, ref_makespan=cm, optimal=opt, seconds=secs, lb=inst.lower_bound())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=6.0)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    (ROOT / "data/ref").mkdir(parents=True, exist_ok=True)
    for split, base, idx in (("val", 5000, 3), ("train", 1000, 2)):
        tasks, s = [], base
        for row in PLAN:
            for _ in range(row[idx]):
                tasks.append((row[0], row[1], s)); s += 1
        t0 = time.time()
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(work)(n, m, sd, a.time_limit) for n, m, sd in tasks)
        with open(ROOT / f"data/ref/ref_{split}.pkl", "wb") as f:
            pickle.dump(res, f)
        opt = sum(r["optimal"] for r in res)
        print(f"{split}: {len(res)} instancias, {opt} óptimas probadas, {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
