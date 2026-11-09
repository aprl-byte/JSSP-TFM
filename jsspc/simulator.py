"""Simulador de despacho basado en el procedimiento de Giffler y Thompson (1960).

El simulador es el mismo para todos los métodos: lo único que cambia entre ellos es la
función `chooser`, que decide qué operación del conjunto de conflicto se despacha.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
from .instances import Instance

RULES = ["FIFO", "SPT", "LPT", "MWKR", "LWKR", "MOPNR"]


class State:
    """Estado mutable de una simulación (se pasa a las reglas y a la extracción de features)."""
    __slots__ = ("inst", "n", "m", "nxt", "jr", "mr", "rem_work", "mach_pending",
                 "done", "total", "start", "mean_p", "max_p", "lb", "ms", "est_star", "ect_star")

    def __init__(self, inst: Instance):
        self.inst = inst
        self.n, self.m = inst.n, inst.m
        self.nxt = np.zeros(self.n, dtype=np.int64)       # índice de la siguiente operación de cada trabajo
        self.jr = np.zeros(self.n)                        # instante en que cada trabajo queda libre
        self.mr = np.zeros(self.m)                        # instante en que cada máquina queda libre
        self.rem_work = inst.times.sum(1).astype(float)   # trabajo restante de cada trabajo
        self.mach_pending = np.array([inst.times[inst.machines == k].sum() for k in range(self.m)], float)
        self.done = 0
        self.total = self.n * self.m
        self.start = np.zeros((self.n, self.m))
        self.mean_p = float(inst.times.mean())
        self.max_p = float(inst.times.max())
        self.lb = float(inst.lower_bound())
        self.ms = 0
        self.est_star = 0.0
        self.ect_star = 0.0


# ----------------------------------------------------------------------------------------------
# Reglas clásicas. Reciben el estado y el conjunto de conflicto C (índices de trabajo, ordenados
# de menor a mayor) y devuelven la posición elegida dentro de C. Desempate: menor índice de trabajo.
# ----------------------------------------------------------------------------------------------
def r_fifo(st, C, p, est):  return int(np.argmin(st.jr[C]))            # llegada a la cola de la máquina
def r_fifo_est(st, C, p, est): return int(np.argmin(est))               # variante del piloto (inicio posible)
def r_spt(st, C, p, est):   return int(np.argmin(p))
def r_lpt(st, C, p, est):   return int(np.argmax(p))
def r_mwkr(st, C, p, est):  return int(np.argmax(st.rem_work[C]))
def r_lwkr(st, C, p, est):  return int(np.argmin(st.rem_work[C]))
def r_mopnr(st, C, p, est): return int(np.argmax(st.m - st.nxt[C]))

CLASSIC = {"FIFO": r_fifo, "SPT": r_spt, "LPT": r_lpt, "MWKR": r_mwkr, "LWKR": r_lwkr, "MOPNR": r_mopnr,
           "FIFO_est": r_fifo_est}


def make_random_rule(seed: int):
    rng = np.random.default_rng(seed)
    return lambda st, C, p, est: int(rng.integers(len(C)))


@dataclass
class Result:
    makespan: float
    start: np.ndarray
    seconds: float
    decisions: int          # nº de decisiones con conflicto real (|C| >= 2)
    steps: int


def run(inst: Instance, chooser, on_decision=None) -> Result:
    """Construye un plan completo.

    chooser(st, C, p, est) -> posición dentro de C.
    on_decision(st, C, p, est, pos) se llama tras cada decisión con conflicto (para etiquetar).
    """
    M, P = inst.machines, inst.times
    st = State(inst)
    n, m = st.n, st.m
    nxt, jr, mr = st.nxt, st.jr, st.mr
    ar = np.arange(n)
    decisions = 0
    t0 = time.perf_counter()
    for _ in range(st.total):
        act = ar[nxt < m]
        k = nxt[act]
        mm = M[act, k]
        pp = P[act, k]
        est_all = np.maximum(jr[act], mr[mm])
        ect_all = est_all + pp
        j = int(np.argmin(ect_all))
        ms = mm[j]
        ect_star = ect_all[j]
        mask = (mm == ms) & (est_all < ect_star)
        mask[j] = True      # con duración 0 la propia operación j no cumple est < ect
        C, p, est = act[mask], pp[mask], est_all[mask]
        st.ms, st.ect_star = int(ms), float(ect_star)
        if len(C) > 1:
            pos = chooser(st, C, p, est)
            decisions += 1
            if on_decision is not None:
                on_decision(st, C, p, est, pos)
        else:
            pos = 0
        i, pi, si = int(C[pos]), int(p[pos]), float(est[pos])
        st.start[i, nxt[i]] = si
        jr[i] = mr[ms] = si + pi
        st.rem_work[i] -= pi
        st.mach_pending[ms] -= pi
        nxt[i] += 1
        st.done += 1
    secs = time.perf_counter() - t0
    return Result(float(jr.max()), st.start, secs, decisions, st.total)


def schedule_metrics(inst: Instance, start: np.ndarray) -> dict:
    """Métricas intermedias de un plan: utilización, tiempo de paso y espera por estación."""
    P, M = inst.times, inst.machines
    end = start + P
    cmax = end.max()
    ready = np.zeros_like(start)
    ready[:, 1:] = end[:, :-1]
    wait = start - ready
    wait_by_machine = np.array([wait[M == k].mean() for k in range(inst.m)])
    return {
        "utilization": float(P.sum() / (inst.m * cmax)),
        "mean_flow": float(end[:, -1].mean()),
        "mean_wait_op": float(wait.mean()),
        "max_wait_machine": float(wait_by_machine.max()),
    }


def validate(inst: Instance, start: np.ndarray) -> bool:
    """Comprueba que un plan es factible (precedencias del trabajo y no solape en máquina)."""
    P, M = inst.times, inst.machines
    end = start + P
    if np.any(start[:, 1:] < end[:, :-1] - 1e-9) or np.any(start < -1e-9):
        return False
    for k in range(inst.m):
        sel = (M == k)
        s, e = start[sel], end[sel]
        o = np.argsort(s)
        if np.any(s[o][1:] < e[o][:-1] - 1e-9):
            return False
    return True
