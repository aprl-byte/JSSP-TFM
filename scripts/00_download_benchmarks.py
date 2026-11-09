"""Descarga los benchmarks públicos (no se redistribuyen en el repositorio).

Fuentes:
  - JSPLIB (Tamy0612): https://github.com/tamy0612/JSPLIB  -> ft, abz, orb, la, swv, yn, ta + óptimo/cotas
  - Corsini et al. (2024): https://github.com/AndreaCorsini1/SelfLabelingJobShop -> DMU (80) con UB
  - OR-Library (Beasley, 1990): jobshop1.txt, 82 instancias (se usa para contrastar con JSPLIB)
Escribe data/raw/<familia>/<nombre>.jsp y data/bks.json con {nombre: [mejor_conocido, es_optimo]}.
"""
import json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
JSPLIB = "https://raw.githubusercontent.com/tamy0612/JSPLIB/master/"
CORSINI = "https://raw.githubusercontent.com/AndreaCorsini1/SelfLabelingJobShop/main/benchmarks/DMU/"
ORLIB = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def family_of(name: str) -> str:
    for f in ("abz", "ft", "orb", "la", "swv", "yn", "ta"):
        if name.startswith(f):
            return f
    raise ValueError(name)


def main():
    meta = json.loads(get(JSPLIB + "instances.json"))
    bks = {}
    for e in meta:
        name, fam = e["name"], family_of(e["name"])
        out = RAW / fam / f"{name}.jsp"
        out.parent.mkdir(parents=True, exist_ok=True)
        if e.get("optimum") is None and e.get("bounds") is None:
            # ta71-ta80 no traen cotas en JSPLIB: se usan fichero y UB del repositorio de Corsini et al.
            if not out.exists():
                out.write_bytes(get(CORSINI.replace("DMU", "TA") + f"{name}.jsp"))
            lines = [l for l in out.read_text().splitlines() if l.strip()]
            bks[name] = [float(lines[int(lines[0].split()[0]) + 1]), False]
            continue
        if not out.exists():
            out.write_bytes(get(JSPLIB + e["path"]))
        if e.get("optimum") is not None:
            bks[name] = [e["optimum"], True]
        else:
            bks[name] = [e["bounds"]["upper"], False]
    # DMU: las cotas superiores vienen en el propio fichero (línea posterior a las n líneas de trabajos)
    for k in range(1, 81):
        name = f"dmu{k:02d}"
        out = RAW / "dmu" / f"{name}.jsp"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.write_bytes(get(CORSINI + f"{name}.jsp"))
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        n = int(lines[0].split()[0])
        bks[name] = [float(lines[n + 1]), False]
    (ROOT / "data" / "bks.json").write_text(json.dumps(bks, indent=0))
    try:
        (RAW / "orlib").mkdir(parents=True, exist_ok=True)
        (RAW / "orlib" / "jobshop1.txt").write_bytes(get(ORLIB))
    except Exception as ex:  # el servidor de Brunel a veces no responde
        print("AVISO: no se pudo descargar OR-Library:", ex)
    print(f"{len(bks)} instancias con mejor conocido -> data/bks.json")


if __name__ == "__main__":
    sys.exit(main())
