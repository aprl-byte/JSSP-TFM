"""Agrega resultados en tablas (results/tables.json) y genera las figuras (figures/*.png)."""
import json, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import wilcoxon
import joblib
from _common import ROOT
from jsspc.instances import read_instance, load_bks
from jsspc.simulator import run, CLASSIC, schedule_metrics
from jsspc.models import make_chooser
from jsspc.reference import solve_cpsat

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.grid": True, "grid.alpha": .25})
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "results/eval_instances.csv")
tr = json.loads((ROOT / "results/training.json").read_text())
T = {}

ORDER = ["FIFO", "SPT", "LPT", "MWKR", "LWKR", "MOPNR", "RANDOM", "LR", "RF", "LGBM", "ENS", "HYB"]
CLASSICS = ["FIFO", "SPT", "LPT", "MWKR", "LWKR", "MOPNR"]
LEARNED = ["LR", "RF", "LGBM", "ENS"]
COL = {m: c for m, c in zip(ORDER, ["#8c8c8c", "#b0b0b0", "#cfcfcf", "#4c78a8", "#9ecae1", "#2b5d8a", "#d9d9d9",
                                    "#e07b39", "#f2b177", "#c0392b", "#8e44ad", "#16a085"])}
# RANDOM: media de las 5 semillas por instancia
rnd = df[df.method == "RANDOM"].groupby(["instance", "family", "size", "n", "m", "bks"], as_index=False).agg(
    makespan=("makespan", "mean"), seconds=("seconds", "mean"), decisions=("decisions", "mean"),
    gap_pct=("gap_pct", "mean"), utilization=("utilization", "mean"), mean_flow=("mean_flow", "mean"),
    mean_wait_op=("mean_wait_op", "mean"), max_wait_machine=("max_wait_machine", "mean"))
rnd["method"] = "RANDOM"
d = pd.concat([df[df.method != "RANDOM"], rnd], ignore_index=True)
d["flow_rel"] = d.mean_flow / d.bks; d["wait_rel"] = d.mean_wait_op / d.bks; d["maxwait_rel"] = d.max_wait_machine / d.bks
d["size_class"] = pd.cut(d.n * d.m, [0, 100, 225, 400, 1000, 3000],
                         labels=["≤100 op.", "101-225", "226-400", "401-1000", ">1000"])
d.to_csv(ROOT / "results/eval_instances_agg.csv", index=False)

# --- Tabla: gap medio por conjunto y método
g = d.pivot_table(index="method", columns="family", values="gap_pct", aggfunc="mean").reindex(ORDER)
g["TODAS"] = d.pivot_table(index="method", values="gap_pct", aggfunc="mean").reindex(ORDER)
med = d.pivot_table(index="method", columns="family", values="gap_pct", aggfunc="median").reindex(ORDER)
worst = d.pivot_table(index="method", columns="family", values="gap_pct", aggfunc="max").reindex(ORDER)
T["gap_mean_by_family"] = g.round(2).reset_index().to_dict("list")
T["gap_median_by_family"] = med.round(2).reset_index().to_dict("list")
T["gap_worst_by_family"] = worst.round(2).reset_index().to_dict("list")
gs = d.pivot_table(index="method", columns="size_class", values="gap_pct", aggfunc="mean", observed=True).reindex(ORDER)
T["gap_mean_by_sizeclass"] = gs.round(2).reset_index().to_dict("list")
T["n_by_sizeclass"] = d[d.method == "FIFO"].groupby("size_class", observed=True).size().to_dict()
T["n_by_family"] = d[d.method == "FIFO"].groupby("family").size().to_dict()

# --- Tiempos: por instancia (s) por clase de tamaño y µs por decisión
tm = d.pivot_table(index="method", columns="size_class", values="seconds", aggfunc="mean", observed=True).reindex(ORDER)
T["seconds_by_sizeclass"] = tm.round(4).reset_index().to_dict("list")
dd = d.assign(us_dec=1e6 * d.seconds / d.decisions.replace(0, np.nan))
T["us_per_decision"] = dd.groupby("method").us_dec.median().reindex(ORDER).round(1).to_dict()
T["total_seconds"] = d.groupby("method").seconds.sum().reindex(ORDER).round(1).to_dict()

# --- Métricas intermedias
im = d.pivot_table(index="method", values=["utilization", "flow_rel", "wait_rel", "maxwait_rel"], aggfunc="mean")
T["intermediate"] = im.reindex(ORDER).round(3).reset_index().to_dict("list")

# --- Victorias y Wilcoxon frente a la mejor clásica (por conjunto y global)
piv = d.pivot_table(index=["instance", "family"], columns="method", values="makespan")
piv = piv.reset_index()
best_classic = {fam: min(CLASSICS, key=lambda m: d[(d.method == m) & (d.family == fam)].gap_pct.mean())
                for fam in ["ORLIB", "TA", "DMU"]}
T["best_classic_by_family"] = best_classic
piv["best_classic_inst"] = piv[CLASSICS].min(axis=1)
rows = []
for m in LEARNED + ["HYB", "RANDOM"]:
    for fam in ["ORLIB", "TA", "DMU", "TODAS"]:
        s = piv if fam == "TODAS" else piv[piv.family == fam]
        bc = best_classic.get(fam) if fam != "TODAS" else min(CLASSICS, key=lambda m: d[d.method == m].gap_pct.mean())
        x, y = s[m].values, s[bc].values
        try:
            p = wilcoxon(x, y, alternative="less").pvalue
        except ValueError:
            p = np.nan
        rows.append(dict(method=m, set=fam, vs=bc, wins=float(100 * np.mean(x < y)), ties=float(100 * np.mean(x == y)),
                         losses=float(100 * np.mean(x > y)), wins_vs_best_per_instance=float(100 * np.mean(x < s["best_classic_inst"].values)),
                         p_wilcoxon=float(p), mean_diff_pct=float(np.mean(100 * (x - y) / y))))
T["wins"] = rows
# Cartera: mejor entre ENS (o LGBM) y MOPNR
port = d.pivot_table(index=["instance", "family", "size_class"], columns="method", values="makespan", observed=True).reset_index()
bks = d[d.method == "FIFO"].set_index("instance").bks
for ml in ("ENS", "LGBM"):
    port[f"PORT_{ml}"] = np.minimum(port[ml], port["MOPNR"])
    port[f"gap_PORT_{ml}"] = 100 * (port[f"PORT_{ml}"] / port.instance.map(bks) - 1)
T["portfolio"] = {ml: {"gap_mean": float(port[f"gap_PORT_{ml}"].mean()),
                       "by_family": port.groupby("family")[f"gap_PORT_{ml}"].mean().round(2).to_dict(),
                       "seconds_sum_mean": float((d[d.method == ml].set_index("instance").seconds +
                                                  d[d.method == "MOPNR"].set_index("instance").seconds).mean())}
                  for ml in ("ENS", "LGBM")}
port["PORT_LGBM_LR"] = np.minimum(port["LGBM"], port["LR"])
port["gap_PORT_LGBM_LR"] = 100 * (port["PORT_LGBM_LR"] / port.instance.map(bks) - 1)
secs = d.pivot_table(index="instance", columns="method", values="seconds")
T["portfolio"]["LGBM+LR"] = {"gap_mean": float(port["gap_PORT_LGBM_LR"].mean()),
                             "by_family": port.groupby("family")["gap_PORT_LGBM_LR"].mean().round(2).to_dict(),
                             "seconds_sum_mean": float((secs["LGBM"] + secs["LR"]).mean())}
T["mean_seconds"] = d.groupby("method").seconds.mean().reindex(ORDER).round(4).to_dict()
# Mejor clásica "oráculo" por instancia (cota de lo que lograría un selector perfecto entre clásicas)
oracle = 100 * (piv["best_classic_inst"].values / piv.instance.map(bks).values - 1)
T["oracle_classic_gap"] = float(np.mean(oracle))

# --- CP-SAT (si existe)
try:
    cs = pd.read_csv(ROOT / "results/eval_cpsat.csv")
    T["cpsat"] = {"n": int(len(cs)), "gap_mean": float(cs.gap_pct.mean()), "optimal_pct": float(100 * cs.optimal.mean()),
                  "seconds_mean": float(cs.seconds.mean()),
                  "by_family": cs.groupby("family").agg(n=("gap_pct", "size"), gap=("gap_pct", "mean"), sec=("seconds", "mean")).round(3).reset_index().to_dict("list")}
    sub = d[d.instance.isin(cs.instance)]
    T["cpsat"]["others_on_subset"] = sub.groupby("method").gap_pct.mean().reindex(ORDER).round(2).to_dict()
except FileNotFoundError:
    cs = None

# --- FT06 con todos los métodos (Tabla / Gantt)
ft = read_instance(ROOT / "data/raw/ft/ft06.jsp", "ft06"); ft.bks = 55
plans, ftrows = {}, []
for r in CLASSICS:
    res = run(ft, CLASSIC[r]); plans[r] = res.start
    ftrows.append(dict(method=r, makespan=res.makespan, **schedule_metrics(ft, res.start), seconds=res.seconds))
for r in LEARNED:
    mdl = joblib.load(ROOT / f"models/{r}.joblib")
    res = run(ft, make_chooser(mdl)); plans[r] = res.start
    ftrows.append(dict(method=r, makespan=res.makespan, **schedule_metrics(ft, res.start), seconds=res.seconds))
cm, opt, secs, st = solve_cpsat(ft, 10, 2); plans["CP-SAT"] = st
ftrows.append(dict(method="CP-SAT", makespan=cm, **schedule_metrics(ft, st), seconds=secs))
T["ft06"] = ftrows

# =================================== FIGURAS ====================================
# 1) Flujo del método
fig, ax = plt.subplots(figsize=(9.2, 2.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
boxes = [("Instancias\nsintéticas\n(semilla fija)", 2), ("Soluciones de\nreferencia\n(CP-SAT)", 21),
         ("Decisiones\netiquetadas\n+ 14 features", 40), ("Clasificador\n(LR / RF /\nLightGBM)", 59), ("Simulador\nGiffler-Thompson\n+ regla aprendida", 78)]
for t, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 9), 18, 14, boxstyle="round,pad=0.4", fc="#e8eef5", ec="#2b5d8a"))
    ax.text(x + 9, 16, t, ha="center", va="center", fontsize=8)
for x in (20.6, 39.6, 58.6, 77.6):
    ax.add_patch(FancyArrowPatch((x, 16), (x + 1.8, 16), arrowstyle="-|>", mutation_scale=10, color="#2b5d8a"))
ax.text(50, 3, "Evaluación: benchmarks no vistos (OR-Library, Taillard, DMU)  ·  makespan, gap, tiempo", ha="center", fontsize=8.5)
ax.add_patch(FancyArrowPatch((87, 8.5), (87, 5.5), arrowstyle="-|>", mutation_scale=10, color="#2b5d8a"))
fig.savefig(FIG / "fig_flujo.png", bbox_inches="tight"); plt.close(fig)

# 2) Gantt FT06
names = ["FIFO", "SPT", "MWKR", "MOPNR", "LGBM", "CP-SAT"]
fig, axs = plt.subplots(3, 2, figsize=(9.2, 7.4), sharex=True)
cmap = plt.get_cmap("tab10")
for ax, nme in zip(axs.ravel(), names):
    s = plans[nme]
    for j in range(ft.n):
        for k in range(ft.m):
            mk = ft.machines[j, k]
            ax.barh(mk, ft.times[j, k], left=s[j, k], color=cmap(j), edgecolor="white", height=.8)
            ax.text(s[j, k] + ft.times[j, k] / 2, mk, str(j + 1), ha="center", va="center", fontsize=6.5, color="white")
    cmx = (s + ft.times).max()
    ax.axvline(cmx, color="k", lw=.8, ls="--")
    ax.set_title(f"{nme}: makespan {cmx:.0f}", fontsize=9); ax.set_yticks(range(ft.m)); ax.set_yticklabels([f"M{k+1}" for k in range(ft.m)])
    ax.invert_yaxis(); ax.grid(axis="y", alpha=0)
for ax in axs[-1]: ax.set_xlabel("Tiempo")
fig.tight_layout(); fig.savefig(FIG / "fig_ft06_gantt.png"); plt.close(fig)

# 3) Convergencia LightGBM
fig, ax = plt.subplots(figsize=(5.6, 3.2))
c = tr["lgbm_curve"]
ax.plot(c["train"], label="Entrenamiento"); ax.plot(c["val"], label="Validación")
ax.axvline(tr["lgbm_best_iteration"], color="k", ls=":", label=f"Parada ({tr['lgbm_best_iteration']} árboles)")
ax.set_xlabel("Iteración (árboles)"); ax.set_ylabel("Pérdida logística"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig_convergencia.png"); plt.close(fig)

# 4) Curva de aprendizaje
lc = pd.DataFrame(tr["learning_curve"])
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
for m, c_ in zip(["LR", "RF", "LGBM"], ["#4c78a8", "#e07b39", "#c0392b"]):
    s = lc[lc.model == m]
    axs[0].plot(s.n_instances, 100 * s.train_acc, "--", color=c_); axs[0].plot(s.n_instances, 100 * s.val_acc, "-o", color=c_, label=m, ms=3)
axs[0].set_xscale("log"); axs[0].set_xlabel("Instancias de entrenamiento"); axs[0].set_ylabel("Acierto top-1 (%)")
axs[0].set_title("Entrenamiento (discontinua) y validación (continua)", fontsize=8.5); axs[0].legend()
s = lc[lc.model == "LGBM"]
axs[1].plot(s.n_instances, s.val_gap_pct, "-o", color="#c0392b", ms=3)
axs[1].axhline(tr["classic_val_gap"]["MOPNR"], color="#2b5d8a", ls="--", label="MOPNR (mejor clásica)")
axs[1].set_xscale("log"); axs[1].set_xlabel("Instancias de entrenamiento"); axs[1].set_ylabel("Gap medio en validación (%)"); axs[1].legend()
axs[1].set_title("LightGBM: makespan en validación", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "fig_curva_aprendizaje.png"); plt.close(fig)

# 5) Importancia de características
imp = tr["importance"]
o = np.argsort(imp["LGBM_gain"])
fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.barh(np.array(imp["feature"])[o], 100 * np.array(imp["LGBM_gain"])[o], color="#c0392b")
ax.set_xlabel("Importancia por ganancia (%)"); fig.tight_layout(); fig.savefig(FIG / "fig_importancia.png"); plt.close(fig)

# 6) Gap por familia y método (cajas)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=False)
show = ["FIFO", "MWKR", "MOPNR", "LR", "RF", "LGBM", "ENS"]
for ax, fam, ttl in zip(axs, ["ORLIB", "TA", "DMU"], ["OR-Library (82)", "Taillard (80)", "DMU (80)"]):
    data = [d[(d.method == m) & (d.family == fam)].gap_pct.values for m in show]
    bp = ax.boxplot(data, tick_labels=show, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], show): patch.set_facecolor(COL[m])
    ax.set_title(ttl, fontsize=9); ax.tick_params(axis="x", rotation=45); ax.set_ylabel("Gap respecto al mejor conocido (%)")
fig.tight_layout(); fig.savefig(FIG / "fig_gap_cajas.png"); plt.close(fig)

# 7) Gap medio por clase de tamaño
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for m in ["FIFO", "MWKR", "MOPNR", "LR", "LGBM", "ENS"]:
    ax.plot(gs.columns.astype(str), gs.loc[m].values, "-o", ms=3, label=m, color=COL[m])
ax.set_xlabel("Nº de operaciones (n × m)"); ax.set_ylabel("Gap medio (%)"); ax.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_gap_tamano.png"); plt.close(fig)

# 8) Tiempo frente a calidad
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for m in ORDER:
    ax.scatter(d[d.method == m].seconds.mean() * 1000, d[d.method == m].gap_pct.mean(), color=COL[m], s=45, edgecolor="k", lw=.4)
    ax.annotate(m, (d[d.method == m].seconds.mean() * 1000, d[d.method == m].gap_pct.mean()), xytext=(4, 3), textcoords="offset points", fontsize=8)
if cs is not None:
    ax.scatter(cs.seconds.mean() * 1000, cs.gap_pct.mean(), marker="s", color="k"); ax.annotate("CP-SAT 10 s*", (cs.seconds.mean() * 1000, cs.gap_pct.mean()), xytext=(-60, 6), textcoords="offset points", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Tiempo medio por instancia (ms, escala log.)"); ax.set_ylabel("Gap medio (%)")
fig.tight_layout(); fig.savefig(FIG / "fig_pareto.png"); plt.close(fig)

# 9) Victorias
w = pd.DataFrame(rows); w = w[w.set != "TODAS"]
fig, ax = plt.subplots(figsize=(6.6, 3.3))
meths = ["LR", "RF", "LGBM", "ENS", "HYB"]
x = np.arange(3); wd = .16
for i, m in enumerate(meths):
    ax.bar(x + i * wd, [w[(w.method == m) & (w.set == f)].wins.values[0] for f in ["ORLIB", "TA", "DMU"]], wd, label=m, color=COL[m])
ax.set_xticks(x + 2 * wd); ax.set_xticklabels(["OR-Library", "Taillard", "DMU"]); ax.set_ylabel("% de instancias con menor makespan\nque la mejor regla clásica del conjunto")
ax.axhline(50, color="k", lw=.6, ls=":"); ax.legend(ncol=5, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_victorias.png"); plt.close(fig)

# 10) Cadena de decisión: umbral del híbrido
gt = pd.DataFrame(tr["gate"])
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(100 * gt.fallback_share, gt.val_gap_pct, "-o", ms=4, color="#16a085")
ax.axhline(tr["classic_val_gap"]["MOPNR"], color="#2b5d8a", ls="--", label="Solo MOPNR")
ax.set_xlabel("Decisiones delegadas en MOPNR (%)"); ax.set_ylabel("Gap medio en validación (%)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig_hibrido.png"); plt.close(fig)

(ROOT / "results/tables.json").write_text(json.dumps(T, indent=1, default=float))
print("tablas y figuras generadas")
