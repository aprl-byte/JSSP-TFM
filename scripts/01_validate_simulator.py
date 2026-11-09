"""Valida el simulador con FT06: reglas clásicas, comprobación de factibilidad y CP-SAT.

Resultado esperado (Tabla 1 de la primera entrega, con FIFO por llegada a la cola):
  SPT 94, LPT 86, MWKR 67, LWKR 96, MOPNR 60, óptimo CP-SAT 55.
"""
import json
from _common import ROOT
from jsspc.instances import read_instance, load_bks
from jsspc.simulator import run, CLASSIC, validate, schedule_metrics
from jsspc.reference import solve_cpsat
from jsspc.labeling import label_instance

inst = read_instance(ROOT / "data/raw/ft/ft06.jsp", "ft06")
bks = load_bks(ROOT / "data/bks.json")["ft06"][0]
out = {"bks": bks, "rules": {}}
for r, f in CLASSIC.items():
    res = run(inst, f)
    assert validate(inst, res.start), f"plan no factible con {r}"
    out["rules"][r] = {"makespan": res.makespan, "gap_pct": round(100 * (res.makespan / bks - 1), 1),
                       "decisions": res.decisions, **schedule_metrics(inst, res.start)}
    print(f"{r:9s} Cmax={res.makespan:5.0f}  gap={out['rules'][r]['gap_pct']:5.1f} %  decisiones={res.decisions}")
cm, opt, secs, st = solve_cpsat(inst, 10, 2)
assert validate(inst, st)
X, y, d, mk = label_instance(inst, st)
out["cpsat"] = {"makespan": cm, "optimal": opt, "seconds": round(secs, 3), "replay_makespan": mk,
                "n_decisions": int(d.max() + 1), "n_rows": int(len(y))}
print("CP-SAT", out["cpsat"])
assert cm == bks and mk == cm, "el simulador no reproduce la solución de referencia"
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/ft06_validation.json").write_text(json.dumps(out, indent=1))
print("OK")
