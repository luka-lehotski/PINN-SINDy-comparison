"""
generate_datasets.py
Generisanje sintetičkih podataka za prigušeni harmonijski oscilator.

Dataset 1: 1% Gaussov šum, pravilan vremenski razmak
Dataset 2: 10% Gaussov šum, nepravilan vremenski razmak (random uzorkovanje)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

os.makedirs('data', exist_ok=True)

# ---------------------------
# FIZIKA: x'' + 0.5x' + 5x = 0
# Analitičko rešenje za podkritično prigušenje
# ---------------------------
def damped_oscillator(t, y, c=0.5, k=5.0):
    """
    Prigušeni harmonijski oscilator.
    x'' + c*x' + k*x = 0
    Pretvaramo u sistem prvog reda:
    y[0] = x,  y[1] = x'
    y[0]' = y[1]
    y[1]' = -c*y[1] - k*y[0]
    """
    return [y[1], -c * y[1] - k * y[0]]

# Parametri sistema
c_true = 0.5   # koeficijent prigušenja
k_true = 5.0   # krutost opruge
x0 = 2.0       # početni položaj
v0 = 0.0       # početna brzina

# Vremenski interval
t_start, t_end = 0, 20

# ---------------------------
# DATASET 1: Čist signal + 1% šum
# ---------------------------
print("Generisanje Dataset 1 (1% šum, pravilan uzorkovanje)...")

t1 = np.linspace(t_start, t_end, 1000)
sol1 = solve_ivp(
    lambda t, y: damped_oscillator(t, y, c_true, k_true),
    [t_start, t_end], [x0, v0], t_eval=t1, method='RK45'
)
x1_clean = sol1.y[0]
v1_clean = sol1.y[1]

# Dodavanje 1% Gaussovog šuma
noise_level_1 = 0.01
x1_noisy = x1_clean + noise_level_1 * np.std(x1_clean) * np.random.randn(len(t1))
v1_noisy = v1_clean + noise_level_1 * np.std(v1_clean) * np.random.randn(len(t1))

df1 = pd.DataFrame({
    't': t1,
    'x': x1_noisy,
    'v': v1_noisy
})
df1.to_csv('data/dataset_1_clean.csv', index=False)

print(f"  Sačuvano: data/dataset_1_clean.csv ({len(t1)} tačaka)")
print(f"  SNR (x): {10*np.log10(np.var(x1_clean)/np.var(x1_noisy - x1_clean)):.1f} dB")

# ---------------------------
# DATASET 2: 10% šum + NEPRAVILNO uzorkovanje
# ---------------------------
print("Generisanje Dataset 2 (10% šum, nepravilno uzorkovanje)...")

# Prvo generišemo na gustoj mreži
t_dense = np.linspace(t_start, t_end, 2000)
sol_dense = solve_ivp(
    lambda t, y: damped_oscillator(t, y, c_true, k_true),
    [t_start, t_end], [x0, v0], t_eval=t_dense, method='RK45'
)

# Nasumično biramo ~400 tačaka (nepravilno uzorkovanje)
np.random.seed(42)
n_points_2 = 400
indices = np.sort(np.random.choice(len(t_dense), n_points_2, replace=False))
t2 = t_dense[indices]

# Dodajemo i nepravilnost: nasumično pomeranje vremena
t2 = t2 + 0.02 * np.random.randn(n_points_2)
t2 = np.sort(t2)  # sortiramo

# Interpoliramo rešenje na nove vremenske tačke
x2_clean = np.interp(t2, t_dense, sol_dense.y[0])
v2_clean = np.interp(t2, t_dense, sol_dense.y[1])

# Dodavanje 10% Gaussovog šuma
noise_level_2 = 0.10
x2_noisy = x2_clean + noise_level_2 * np.std(x2_clean) * np.random.randn(len(t2))
v2_noisy = v2_clean + noise_level_2 * np.std(v2_clean) * np.random.randn(len(t2))

df2 = pd.DataFrame({
    't': t2,
    'x': x2_noisy,
    'v': v2_noisy
})
df2.to_csv('data/dataset_2_noisy.csv', index=False)

print(f"  Sačuvano: data/dataset_2_noisy.csv ({len(t2)} tačaka)")
print(f"  SNR (x): {10*np.log10(np.var(x2_clean)/np.var(x2_noisy - x2_clean)):.1f} dB")
print(f"  Prosečan dt: {np.mean(np.diff(t2)):.4f}, std dt: {np.std(np.diff(t2)):.4f}")

# ---------------------------
# VIZUALIZACIJA ZA PROVERU
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(t1, x1_clean, 'k-', alpha=0.3, lw=2, label='Čist signal')
axes[0, 0].plot(t1, x1_noisy, 'r.', markersize=2, alpha=0.7, label='Sa 1% šumom')
axes[0, 0].set_title('Dataset 1 - Vremenska serija')
axes[0, 0].set_xlabel('t'); axes[0, 0].set_ylabel('x(t)')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t2, x2_clean, 'k-', alpha=0.3, lw=2, label='Čist signal')
axes[0, 1].scatter(t2, x2_noisy, c='r', s=15, alpha=0.7, label='Sa 10% šumom')
axes[0, 1].set_title('Dataset 2 - Vremenska serija (nepravilno uzorkovan)')
axes[0, 1].set_xlabel('t'); axes[0, 1].set_ylabel('x(t)')
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x1_clean, v1_clean, 'k-', alpha=0.5, lw=2, label='Čist fazni portret')
axes[1, 0].plot(x1_noisy, v1_noisy, 'r.', markersize=2, alpha=0.5, label='Sa šumom')
axes[1, 0].set_title('Dataset 1 - Fazni dijagram')
axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('v')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

axes[1, 1].plot(x2_clean, v2_clean, 'k-', alpha=0.5, lw=2, label='Čist fazni portret')
axes[1, 1].scatter(x2_noisy, v2_noisy, c='r', s=15, alpha=0.5, label='Sa šumom')
axes[1, 1].set_title('Dataset 2 - Fazni dijagram')
axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('v')
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axis('equal')

plt.tight_layout()
plt.savefig('data/datasets_overview.png', dpi=150)
plt.show()

print("\nGenerisanje završeno!")
print(f"Dataset 1: data/dataset_1_clean.csv")
print(f"Dataset 2: data/dataset_2_noisy.csv")