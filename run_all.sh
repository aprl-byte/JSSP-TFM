#!/usr/bin/env bash
# Reproduce todos los resultados de la memoria (algo menos de 1 hora en un equipo de 2 núcleos).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python -m pytest -q tests
python scripts/00_download_benchmarks.py
python scripts/01_validate_simulator.py
python scripts/02_generate_reference.py --time-limit 6 --jobs 2   | tee logs/02_generate_reference.log
python scripts/03_build_decisions.py
python scripts/04_train.py                                          | tee logs/04_train.log
python scripts/05_evaluate.py --jobs 2                              | tee logs/05_evaluate.log
python scripts/05_evaluate.py --cpsat                               | tee logs/05_evaluate_cpsat.log
python scripts/06_summarize.py
