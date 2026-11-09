"""Lectura y generación de instancias del Job-Shop Scheduling (JSSP)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass
class Instance:
    name: str
    machines: np.ndarray          # (n, m) máquina de la k-ésima operación de cada trabajo
    times: np.ndarray             # (n, m) duración de cada operación
    bks: float | None = None      # mejor solución conocida (óptimo o cota superior)
    bks_optimal: bool = False
    family: str = "synthetic"

    @property
    def n(self) -> int:
        return self.machines.shape[0]

    @property
    def m(self) -> int:
        return self.machines.shape[1]

    @property
    def size(self) -> str:
        return f"{self.n}x{self.m}"

    def lower_bound(self) -> int:
        """Cota inferior trivial: máx(trabajo más largo, máquina más cargada)."""
        job = self.times.sum(1).max()
        mach = max(self.times[self.machines == k].sum() for k in range(self.m))
        return int(max(job, mach))


def read_instance(path: str | Path, name: str | None = None, family: str = "") -> Instance:
    """Lee el formato estándar 'n m' + n líneas de pares (máquina, tiempo).

    Es válido tanto para JSPLIB como para los ficheros .jsp de Corsini et al.
    (que añaden después una línea con la cota superior y una solución, que se ignoran).
    """
    path = Path(path)
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    first = lines[0].split()
    n, m = int(first[0]), int(first[1])
    mach = np.zeros((n, m), dtype=np.int64)
    tim = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        v = list(map(int, lines[1 + i].split()))
        mach[i] = v[0:2 * m:2]
        tim[i] = v[1:2 * m:2]
    inst = Instance(name or path.stem, mach, tim, family=family)
    # Los .jsp de Corsini et al. añaden, tras las n líneas, una línea con la cota superior (UB)
    if len(lines) > n + 1 and len(lines[n + 1].split()) == 1:
        inst.bks = float(lines[n + 1])
    return inst


def read_orlib(path: str | Path) -> list[Instance]:
    """Lee el fichero jobshop1.txt de la OR-Library (82 instancias en un único fichero)."""
    txt = Path(path).read_text().splitlines()
    out, i = [], 0
    while i < len(txt):
        if txt[i].strip().startswith("instance"):
            name = txt[i].split()[1]
            # saltar la descripción hasta la línea con 'n m'
            j = i + 1
            while True:
                tok = txt[j].split()
                if len(tok) == 2 and all(t.isdigit() for t in tok) and j > i + 1:
                    break
                j += 1
            n, m = map(int, txt[j].split())
            mach = np.zeros((n, m), dtype=np.int64)
            tim = np.zeros((n, m), dtype=np.int64)
            for r in range(n):
                v = list(map(int, txt[j + 1 + r].split()))
                mach[r], tim[r] = v[0:2 * m:2], v[1:2 * m:2]
            out.append(Instance(name, mach, tim, family="ORLIB"))
            i = j + n + 1
        else:
            i += 1
    return out


def generate(n: int, m: int, seed: int, tmin: int = 1, tmax: int = 99, name: str | None = None) -> Instance:
    """Instancia sintética al estilo Taillard: ruta aleatoria por trabajo y duraciones U[1, 99].

    No reproduce el generador congruencial original de Taillard (1993), sino su procedimiento
    con numpy y semilla fija; las instancias NO son las del benchmark oficial.
    """
    rng = np.random.default_rng(seed)
    mach = np.stack([rng.permutation(m) for _ in range(n)])
    tim = rng.integers(tmin, tmax + 1, size=(n, m))
    return Instance(name or f"syn_{n}x{m}_s{seed}", mach.astype(np.int64), tim.astype(np.int64))


def load_bks(path: str | Path) -> dict[str, tuple[float, bool]]:
    """Carga data/bks.json: {nombre: [valor, es_optimo]}."""
    d = json.loads(Path(path).read_text())
    return {k: (float(v[0]), bool(v[1])) for k, v in d.items()}
