"""Evaluación de reglas de despacho sobre conjuntos de instancias."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .simulator import run, schedule_metrics


def eval_rule(insts, chooser_factory, label: str, with_metrics: bool = False) -> pd.DataFrame:
    """Ejecuta una regla sobre todas las instancias. chooser_factory(inst) devuelve el chooser."""
    rows = []
    for inst in insts:
        ch = chooser_factory(inst)
        t0 = time.perf_counter()
        res = run(inst, ch)
        dt = time.perf_counter() - t0
        row = dict(method=label, instance=inst.name, family=inst.family, size=inst.size, n=inst.n, m=inst.m,
                   makespan=res.makespan, bks=inst.bks, seconds=dt, decisions=res.decisions)
        row["gap_pct"] = 100 * (res.makespan / inst.bks - 1) if inst.bks else np.nan
        if with_metrics:
            row.update(schedule_metrics(inst, res.start))
        rows.append(row)
    return pd.DataFrame(rows)
