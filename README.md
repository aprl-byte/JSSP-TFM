# jssp-compare

Comparativa de reglas de despacho clásicas (FIFO, SPT, LPT, MWKR, LWKR, MOPNR) frente a reglas aprendidas con
aprendizaje automático supervisado (regresión logística, Random Forest, LightGBM y su conjunto) en el
job-shop scheduling (JSSP). Código de apoyo del Trabajo Fin de Máster (Máster Universitario en Inteligencia
Artificial, UNIR), segunda entrega.

Todos los métodos se ejecutan en el mismo simulador (procedimiento de Giffler y Thompson), sobre las mismas
instancias y con las mismas métricas. Los benchmarks públicos no se redistribuyen: se descargan con un script.

## Estructura

| Ruta | Contenido |
|---|---|
| `jsspc/` | Paquete: lectura y generación de instancias, simulador, características, etiquetado, referencia CP-SAT, modelos y evaluación |
| `scripts/` | Pasos numerados 00 a 07 (ver más abajo) |
| `tests/` | Pruebas automáticas (valores de FT06 y factibilidad de los planes) |
| `data/ref/` | Soluciones de referencia de CP-SAT (entrenamiento y validación). Se versionan porque CP-SAT con varios hilos no es determinista |
| `results/` | Resultados en CSV y JSON que usa la memoria |
| `figures/` | Figuras de la memoria |
| `pilot/` | Código del experimento preliminar de la primera entrega (FIFO por inicio más temprano, solo para reproducir aquel resultado) |
| `run_all.sh` | Ejecuta todo el flujo de principio a fin |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso rápido

```bash
python -m pytest -q tests                           # comprobaciones del simulador
python scripts/00_download_benchmarks.py            # descarga benchmarks a data/raw (no se versiona)
python scripts/01_validate_simulator.py             # valida el simulador con FT06
python scripts/02_generate_reference.py --time-limit 6 --jobs 2   # instancias sintéticas + CP-SAT
python scripts/03_build_decisions.py                # decisiones etiquetadas
python scripts/04_train.py                          # entrenamiento y diagnóstico
python scripts/05_evaluate.py --jobs 2              # evaluación en OR-Library, Taillard y DMU
python scripts/05_evaluate.py --cpsat               # comparación con CP-SAT (subconjunto de 68 instancias)
python scripts/06_summarize.py                      # tablas y figuras
python scripts/07_case_study.py                     # caso de aplicación ilustrativo
```

También puede ejecutarse todo con `bash run_all.sh` (algo menos de 1 hora en 2 núcleos). Los tiempos dependen de la máquina.

## Valores de control (FT06)

FIFO 65, SPT 94, LPT 86, MWKR 67, LWKR 96, MOPNR 60 y óptimo CP-SAT 55.

## Reproducibilidad

- Semillas fijas para generación de instancias (entrenamiento 1000+, validación 5000+), modelos y regla aleatoria (5 semillas).
- Las soluciones de referencia están en `data/ref/`; regenerarlas puede dar planes distintos con el mismo makespan o muy parecido.
- Los modelos entrenados (`models/`) y las decisiones (`data/decisions_*.npz`) no se versionan: se regeneran con los pasos 03 y 04.

## Licencia y datos

Los benchmarks pertenecen a sus autores (Beasley, Fisher y Thompson, Taillard, Demirkol et al., Adams et al., entre otros) y se
descargan de JSPLIB y del repositorio de Corsini et al. (2024). El modelo preentrenado de Corsini et al. no se incluye.
