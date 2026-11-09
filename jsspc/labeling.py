"""Etiquetado de decisiones a partir de una solución de referencia."""
from __future__ import annotations
import numpy as np
from .instances import Instance
from .simulator import run
from .features import extract, NF


def label_instance(inst: Instance, ref_start: np.ndarray):
    """Reproduce la solución de referencia en el simulador y etiqueta cada decisión.

    En cada paso con conflicto real (|C| >= 2) la candidata positiva es la operación del conjunto
    de conflicto que la solución de referencia programa antes en esa máquina; las demás son negativas.

    Devuelve (X, y, decision_id, makespan_replay). makespan_replay es el makespan que obtiene el
    simulador al seguir la referencia; debe coincidir con el de la referencia (comprobación de validez).
    """
    rows_X, rows_y, rows_d = [], [], []
    counter = [0]

    def chooser(st, C, p, est):
        ref = ref_start[C, st.nxt[C]]
        return int(np.argmin(ref))          # desempate: menor índice de trabajo

    def on_decision(st, C, p, est, pos):
        # st.nxt todavía no se ha actualizado: las features describen el estado previo a la decisión
        X = extract(st, C, p, est)
        y = np.zeros(len(C)); y[pos] = 1
        rows_X.append(X); rows_y.append(y)
        rows_d.append(np.full(len(C), counter[0])); counter[0] += 1

    res = run(inst, chooser, on_decision)
    if not rows_X:
        return np.empty((0, NF)), np.empty(0), np.empty(0, int), res.makespan
    return np.vstack(rows_X), np.concatenate(rows_y), np.concatenate(rows_d), res.makespan
