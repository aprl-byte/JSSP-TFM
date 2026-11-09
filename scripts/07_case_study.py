"""Caso de aplicación ilustrativo: secuenciación de pacientes en un bloque quirúrgico.

Cada paciente es un trabajo y cada recurso una estación (máquina). Las duraciones son SINTÉTICAS
(minutos, órdenes de magnitud plausibles); el caso solo demuestra que el mismo simulador y las mismas
reglas se aplican a un problema con estructura de job-shop. Una duración 0 indica que el paciente no usa esa estación.
Basado en la formulación de Pham y Klinkert (2008): cirugía como job-shop generalizado.
"""
import json
import numpy as np, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import ROOT
from jsspc.instances import Instance
from jsspc.simulator import run, CLASSIC, schedule_metrics
from jsspc.models import make_chooser
from jsspc.reference import solve_cpsat

STATIONS = ["Preparación", "Imagen", "Quirófano A", "Quirófano B", "Reanimación"]
rng = np.random.default_rng(2026)
n = 10
mach = np.zeros((n, 5), dtype=int); tim = np.zeros((n, 5), dtype=int)
for i in range(n):
    needs_img = rng.random() < 0.6
    room = 2 if rng.random() < 0.5 else 3
    other = 5 - room
    mid = [1, room, other]; 
    # el orden de las estaciones intermedias varía: algunos pacientes pasan antes por imagen, otros después
    if needs_img and rng.random() < 0.5: mid = [room, 1, other]
    elif rng.random() < 0.5: mid = [room, other, 1]
    route = [0] + mid + [4]
    mach[i] = route
    dur = {0: rng.integers(10, 31), 1: rng.integers(10, 26) if needs_img else 0,
           room: rng.integers(45, 151), other: 0, 4: rng.integers(30, 91)}
    tim[i] = [dur[r] for r in route]
inst = Instance("quirofano_10", mach, tim)
inst.bks = None
out = {"durations_min": tim.tolist(), "routes": mach.tolist(), "methods": {}}
plans = {}
for name in ["FIFO", "SPT", "MWKR", "MOPNR"]:
    r = run(inst, CLASSIC[name]); plans[name] = r.start
    out["methods"][name] = dict(makespan=r.makespan, seconds=r.seconds, **schedule_metrics(inst, r.start))
for name in ["LR", "LGBM", "ENS"]:
    mdl = joblib.load(ROOT / f"models/{name}.joblib")
    r = run(inst, make_chooser(mdl)); plans[name] = r.start
    out["methods"][name] = dict(makespan=r.makespan, seconds=r.seconds, **schedule_metrics(inst, r.start))
cm, opt, secs, st = solve_cpsat(inst, 10, 2); plans["CP-SAT"] = st
out["methods"]["CP-SAT"] = dict(makespan=cm, seconds=secs, optimal=opt, **schedule_metrics(inst, st))
out["lower_bound"] = inst.lower_bound()
for k, v in out["methods"].items():
    print(f"{k:7s} Cmax={v['makespan']:.0f} min  util={v['utilization']:.2f}  espera media/op={v['mean_wait_op']:.1f}")
(ROOT / "results/case_study.json").write_text(json.dumps(out, indent=1, default=float))

fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.4), sharex=True)
cmap = plt.get_cmap("tab10")
for ax, nme in zip(axs, ["MOPNR", "CP-SAT"]):
    s = plans[nme]
    for j in range(n):
        for k in range(5):
            if tim[j, k] == 0: continue
            ax.barh(mach[j, k], tim[j, k], left=s[j, k], color=cmap(j), edgecolor="white", height=.8)
            ax.text(s[j, k] + tim[j, k] / 2, mach[j, k], f"P{j+1}", ha="center", va="center", fontsize=6, color="white")
    ax.set_yticks(range(5)); ax.set_yticklabels(STATIONS, fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", alpha=0)
    ax.set_title(f"{nme}: {out['methods'][nme]['makespan']:.0f} min", fontsize=9); ax.set_xlabel("Minutos")
fig.tight_layout(); fig.savefig(ROOT / "figures/fig_caso_quirofano.png"); plt.close(fig)
