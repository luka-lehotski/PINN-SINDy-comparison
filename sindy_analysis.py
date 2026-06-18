"""
sindy_model.py
SINDy (Sparse Identification of Nonlinear Dynamics) modul za identifikaciju
parametara prigušenog harmonijskog oscilatora.

Sadrži optimizaciju thresholda kroz unakrsnu validaciju (CV) i simulaciju trajektorija.
"""

import numpy as np
import time
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.optimizers import STLSQ

def _sindy_get_params(model):
    """Izvlači c i k iz SINDy modela na osnovu imena karakteristika."""
    coeffs = model.coefficients()
    if coeffs.shape[0] < 2:
        return np.nan, np.nan
    feat_names = model.get_feature_names()
    try:
        x_idx = feat_names.index('x')
        v_idx = feat_names.index('v')
        return -coeffs[1, v_idx], -coeffs[1, x_idx]
    except ValueError:
        return -coeffs[1, 1], -coeffs[1, 0]

def _sindy_trajectory_mse(model, X, t):
    """Računa MSE između stvarnih podataka i SINDy simulacije sistema."""
    try:
        x_sim = model.simulate(X[0], t=t)
        n = min(len(X), len(x_sim))
        return float(np.mean((X[:n, 0] - x_sim[:n, 0]) ** 2))
    except Exception:
        return np.inf

def sindy_analyze(t, X, dataset_name, noise_level="low", thresholds=(0.01, 0.02, 0.05, 0.1), fixed_threshold=None):
    """
    Glavna funkcija za SINDy analizu. 
    Vrši automatsku selekciju hiperparametara (threshold) ako nije fiksiran.
    """
    dt = np.mean(np.diff(t))
    diff_method = SmoothedFiniteDifference()
    poly_lib = PolynomialLibrary(degree=2, include_bias=True)

    if fixed_threshold is not None:
        best_threshold = fixed_threshold
        cv_results = []
    else:
        # 80/20 unakrsna validacija (CV) za pronalaženje optimalnog lambda/thresholda
        n = len(t)
        n_train = int(0.8 * n)
        t_train, t_val = t[:n_train], t[n_train:]
        X_train, X_val = X[:n_train], X[n_train:]
        dt_train = np.mean(np.diff(t_train))

        cv_results = []
        for thr in thresholds:
            try:
                m = ps.SINDy(differentiation_method=diff_method, feature_library=poly_lib, optimizer=STLSQ(threshold=thr, normalize_columns=True))
                m.fit(X_train, t=dt_train, feature_names=["x", "v"])
                val_mse = _sindy_trajectory_mse(m, X_val, t_val)
                n_terms = int(np.count_nonzero(m.coefficients()))
                cv_results.append({'threshold': thr, 'val_mse': val_mse, 'n_terms': n_terms})
            except Exception:
                cv_results.append({'threshold': thr, 'val_mse': np.inf, 'n_terms': 0})

        valid = [r for r in cv_results if r['val_mse'] < np.inf and r['n_terms'] > 0]
        if not valid:
            best_threshold = 0.05
        else:
            min_mse = min(r['val_mse'] for r in valid)
            acceptable = [r for r in valid if r['val_mse'] <= max(2.0 * min_mse, min_mse + 1e-6)]
            best_threshold = min(acceptable, key=lambda r: (r['n_terms'], r['val_mse']))['threshold']

    # Finalni model sa najboljim thresholdom obučen na kompletnom skupu podataka
    start_time = time.time()
    final_model = ps.SINDy(differentiation_method=diff_method, feature_library=poly_lib, optimizer=STLSQ(threshold=best_threshold, normalize_columns=True))
    final_model.fit(X, t=dt, feature_names=["x", "v"])
    training_time = time.time() - start_time

    c_identified, k_identified = _sindy_get_params(final_model)
    mse = _sindy_trajectory_mse(final_model, X, t)

    try:
        t_dense = np.linspace(t.min(), t.max(), 500)
        x_sim_dense = final_model.simulate(X[0], t=t_dense)
    except Exception:
        t_dense = t
        x_sim_dense = None

    return {
        'model': final_model, 'mse': mse, 'c': c_identified, 'k': k_identified, 
        'time': training_time, 't_dense': t_dense, 'x_sim_dense': x_sim_dense, 
        'best_threshold': best_threshold, 'cv_results': cv_results
    }