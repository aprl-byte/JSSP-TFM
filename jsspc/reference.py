"""Soluciones de referencia con CP-SAT (Google OR-Tools)."""
from __future__ import annotations
import time
import numpy as np
from ortools.sat.python import cp_model
from .instances import Instance


def solve_cpsat(inst: Instance, time_limit: float = 10.0, workers: int = 2, seed: int = 0,
                hint: np.ndarray | None = None):
    """Resuelve el JSSP con CP-SAT. Devuelve (makespan, es_óptimo, segundos, inicios[n,m])."""
    n, m = inst.n, inst.m
    P, M = inst.times, inst.machines
    H = int(P.sum())
    md = cp_model.CpModel()
    S = np.empty((n, m), dtype=object)
    per_machine = [[] for _ in range(m)]
    for i in range(n):
        for k in range(m):
            S[i, k] = md.NewIntVar(0, H, f"s{i}_{k}")
            per_machine[int(M[i, k])].append(
                md.NewFixedSizeIntervalVar(S[i, k], int(P[i, k]), f"i{i}_{k}"))
            if k > 0:
                md.Add(S[i, k] >= S[i, k - 1] + int(P[i, k - 1]))
    for lst in per_machine:
        md.AddNoOverlap(lst)
    cmax = md.NewIntVar(0, H, "cmax")
    md.AddMaxEquality(cmax, [S[i, m - 1] + int(P[i, m - 1]) for i in range(n)])
    md.Minimize(cmax)
    if hint is not None:
        for i in range(n):
            for k in range(m):
                md.AddHint(S[i, k], int(hint[i, k]))
    so = cp_model.CpSolver()
    so.parameters.max_time_in_seconds = float(time_limit)
    so.parameters.num_workers = int(workers)
    so.parameters.random_seed = int(seed)
    t0 = time.perf_counter()
    status = so.Solve(md)
    secs = time.perf_counter() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, False, secs, None
    start = np.array([[so.Value(S[i, k]) for k in range(m)] for i in range(n)], dtype=float)
    return float(so.Value(cmax)), status == cp_model.OPTIMAL, secs, start
