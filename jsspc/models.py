"""Modelos de la regla aprendida y envoltorios para usarlos como regla de despacho."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from .features import extract
from .simulator import CLASSIC


class LRModel:
    name = "LR"
    def __init__(self, C=1.0):
        self.C = C
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
    def fit(self, X, y, **kw):
        self.m.fit(X, y); return self
    def score(self, X):
        # la tubería es lineal: evita la sobrecarga de sklearn en cada decisión
        sc, lr = self.m.steps[0][1], self.m.steps[1][1]
        return ((X - sc.mean_) / sc.scale_) @ lr.coef_[0] + lr.intercept_[0]


class RFModel:
    name = "RF"
    def __init__(self, n_estimators=60, max_depth=12, min_samples_leaf=10, seed=0):
        self.m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf, n_jobs=1, random_state=seed)
    def fit(self, X, y, **kw):
        self.m.fit(X, y); return self
    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class LGBMModel:
    name = "LGBM"
    def __init__(self, num_leaves=31, learning_rate=0.05, n_estimators=600, min_child_samples=40, seed=0):
        self.params = dict(objective="binary", num_leaves=num_leaves, learning_rate=learning_rate,
                           min_child_samples=min_child_samples, feature_fraction=0.9, bagging_fraction=0.8,
                           bagging_freq=1, seed=seed, num_threads=1, verbose=-1)
        self.n_estimators = n_estimators
        self.evals = {}
        self.best_iteration = None
    def fit(self, X, y, Xv=None, yv=None, early_stopping=40):
        dtr = lgb.Dataset(X, y)
        valid, cbs = [dtr], [lgb.record_evaluation(self.evals)]
        names = ["train"]
        if Xv is not None:
            valid.append(lgb.Dataset(Xv, yv, reference=dtr)); names.append("val")
            cbs.append(lgb.early_stopping(early_stopping, verbose=False))
        self.b = lgb.train(self.params, dtr, self.n_estimators, valid_sets=valid, valid_names=names, callbacks=cbs)
        self.best_iteration = self.b.best_iteration or self.n_estimators
        return self
    def score(self, X):
        return self.b.predict(X, num_iteration=self.best_iteration)


class Ensemble:
    """Media de las probabilidades de RF y LightGBM."""
    name = "ENS"
    def __init__(self, rf, gb):
        self.rf, self.gb = rf, gb
    def score(self, X):
        return 0.5 * (self.rf.score(X) + self.gb.score(X))


def _to_prob(s, model):
    return 1 / (1 + np.exp(-s)) if model.name == "LR" else s


def make_chooser(model):
    """Regla de despacho aprendida: elige la candidata con mayor puntuación (empate: menor índice)."""
    def chooser(st, C, p, est):
        return int(np.argmax(model.score(extract(st, C, p, est))))
    return chooser


def make_gated_chooser(model, fallback: str, tau: float):
    """Híbrido con discriminante de confianza: si la diferencia entre las dos mejores
    probabilidades es menor que tau, se usa la regla clásica `fallback`; si no, el modelo."""
    fb = CLASSIC[fallback]
    stats = {"ml": 0, "fallback": 0}
    def chooser(st, C, p, est):
        s = _to_prob(model.score(extract(st, C, p, est)), model)
        top2 = np.partition(s, -2)[-2:]
        if top2[1] - top2[0] < tau:
            stats["fallback"] += 1
            return fb(st, C, p, est)
        stats["ml"] += 1
        return int(np.argmax(s))
    chooser.stats = stats
    return chooser


def top1_accuracy(model, X, y, g):
    """Porcentaje de decisiones en las que la candidata de mayor puntuación coincide con la referencia."""
    s = model.score(X)
    order = np.lexsort((-s, g))                 # por decisión y puntuación descendente
    first = np.r_[True, g[order][1:] != g[order][:-1]]
    return float(y[order][first].mean())
