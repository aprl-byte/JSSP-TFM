"""Características de cada operación candidata en una decisión de despacho.

Todas están normalizadas respecto al conjunto de conflicto o a la instancia, de modo que
no dependen directamente del tamaño ni de la escala de las duraciones. Esto es lo que permite
entrenar con instancias pequeñas y evaluar en instancias mayores (generalización).
"""
from __future__ import annotations
import numpy as np
from .simulator import State

FEATURE_NAMES = [
    "p_rel_C",         # duración / media de duraciones del conjunto de conflicto
    "p_rel_inst",      # duración / duración máxima de la instancia
    "rw_rel_C",        # trabajo restante / máximo trabajo restante del conjunto de conflicto
    "rw_rel_lb",       # trabajo restante / cota inferior del makespan
    "rops_rel",        # operaciones restantes / nº de máquinas
    "wait_rel",        # tiempo que lleva esperando la operación en cola / duración media
    "est_gap_rel",     # (inicio posible - mínimo inicio posible del conjunto) / duración media
    "next_p_rel",      # duración de la operación siguiente del trabajo / duración media
    "next_load_rel",   # carga pendiente de la máquina de la operación siguiente / carga media
    "rank_rw",         # posición (0..1) según trabajo restante dentro del conjunto
    "rank_p",          # posición (0..1) según duración dentro del conjunto
    "is_last",         # 1 si es la última operación del trabajo
    "C_size_rel",      # tamaño del conjunto de conflicto / nº de trabajos
    "progress",        # fracción de operaciones ya despachadas
]
NF = len(FEATURE_NAMES)


def _rank01(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros(len(x))
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / (len(x) - 1)


def extract(st: State, C: np.ndarray, p: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Matriz (|C|, NF) con las características de cada candidata."""
    inst = st.inst
    nxt = st.nxt[C]
    rw = st.rem_work[C]
    mean_p_C = p.mean()
    mean_p = st.mean_p
    wait = np.maximum(0.0, st.mr[st.ms] - st.jr[C])
    has_next = (nxt + 1) < st.m
    nxt_idx = np.minimum(nxt + 1, st.m - 1)
    next_p = np.where(has_next, inst.times[C, nxt_idx], 0.0)
    next_m = inst.machines[C, nxt_idx]
    mean_load = max(st.mach_pending.mean(), 1e-9)
    next_load = np.where(has_next, st.mach_pending[next_m], 0.0)
    X = np.empty((len(C), NF))
    X[:, 0] = p / mean_p_C
    X[:, 1] = p / st.max_p
    X[:, 2] = rw / max(rw.max(), 1e-9)
    X[:, 3] = rw / st.lb
    X[:, 4] = (st.m - nxt) / st.m
    X[:, 5] = wait / mean_p
    X[:, 6] = (est - est.min()) / mean_p
    X[:, 7] = next_p / mean_p
    X[:, 8] = next_load / mean_load
    X[:, 9] = _rank01(rw)
    X[:, 10] = _rank01(p)
    X[:, 11] = (~has_next).astype(float)
    X[:, 12] = len(C) / st.n
    X[:, 13] = st.done / st.total
    return X
