"""
comparison.py (Ultimativna SoC Verzija sa Faznim Prostorom i Parametarskim Bar Plotovima)
Centralno poređenje SINDy i PINN metode za identifikaciju parametara.

Sadrži:
  - Eksterni PINN i SINDy uvoz (Separation of Concerns).
  - Prilagođavanje za nepravilan vremenski uzorak (DS2).
  - Ispravno učitavanje Kaggle dataseta (DS3) sa kolonama time, displacement, velocity.
  - Generisanje odvojenih bar plotova za c i k sa označenom teorijskom vrednošću.
  - Generisanje faznih dijagrama (Phase Space: x vs v) za sve datasete.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy.integrate import solve_ivp

# Uvoz eksternih modela po SoC principu
from pinn_model import DampedPINN
from sindy_model import sindy_analyze

# ---------------------------
# PODEŠAVANJA GRAFIKA
# ---------------------------
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

C_TRUE = 0.5
K_TRUE = 5.0

# ---------------------------
# GENERISANJE SINTETIČKIH PODATAKA
# ---------------------------
def generate_datasets():
    def damped_oscillator(t, y, c=C_TRUE, k=K_TRUE):
        return [y[1], -c * y[1] - k * y[0]]
    
    if not os.path.exists('data/dataset_1_clean.csv'):
        print("Generisanje Dataset 1...")
        t1 = np.linspace(0, 20, 1000)
        sol1 = solve_ivp(lambda t, y: damped_oscillator(t, y), [0, 20], [2.0, 0.0], t_eval=t1)
        x1 = sol1.y[0] + 0.01 * np.std(sol1.y[0]) * np.random.randn(len(t1))
        v1 = sol1.y[1] + 0.01 * np.std(sol1.y[1]) * np.random.randn(len(t1))
        pd.DataFrame({'t': t1, 'x': x1, 'v': v1}).to_csv('data/dataset_1_clean.csv', index=False)
    
    if not os.path.exists('data/dataset_2_noisy.csv'):
        print("Generisanje Dataset 2...")
        t_dense = np.linspace(0, 20, 2000)
        sol_dense = solve_ivp(lambda t, y: damped_oscillator(t, y), [0, 20], [2.0, 0.0], t_eval=t_dense)
        np.random.seed(42)
        indices = np.sort(np.random.choice(len(t_dense), 400, replace=False))
        t2 = t_dense[indices] + 0.02 * np.random.randn(400)
        t2 = np.sort(t2)
        x2 = np.interp(t2, t_dense, sol_dense.y[0])
        v2 = np.interp(t2, t_dense, sol_dense.y[1])
        x2 += 0.10 * np.std(x2) * np.random.randn(len(t2))
        v2 += 0.10 * np.std(v2) * np.random.randn(len(t2))
        pd.DataFrame({'t': t2, 'x': x2, 'v': v2}).to_csv('data/dataset_2_noisy.csv', index=False)

# ---------------------------
# PINN OMOTAČ
# ---------------------------
def pinn_analyze_wrapper(t, x, dataset_key, epochs=5000):
    is_noisy = "DS2" in dataset_key or "DS3" in dataset_key
    pinn = DampedPINN(t, x, epochs=epochs, is_noisy=is_noisy)
    
    start_time = time.time()
    pinn.train(verbose=True) 
    training_time = time.time() - start_time
    
    x_pred = pinn.predict(t)
    mse = np.mean((x - x_pred)**2)
    
    t_dense = np.linspace(t.min(), t.max(), 500)
    x_pred_dense = pinn.predict(t_dense)
    v_pred_dense = np.gradient(x_pred_dense, t_dense)

    return {
        'model': pinn, 'mse': mse, 'c': pinn.c.item(), 'k': pinn.k.item(), 'time': training_time,
        't_dense': t_dense, 'x_pred_dense': x_pred_dense, 'v_pred_dense': v_pred_dense
    }

# ---------------------------
# GLAVNA PETLJA POREĐENJA
# ---------------------------
def run_comparison():
    generate_datasets()
    datasets = {}
    
    if os.path.exists('data/dataset_1_clean.csv'):
        df1 = pd.read_csv('data/dataset_1_clean.csv')
        datasets['DS1'] = {'name': 'Dataset 1 (1% šum, pravilan)', 't': df1['t'].values, 'X': df1[['x', 'v']].values, 'x': df1['x'].values, 'noise': 'low', 'epochs': 5000}
    
    if os.path.exists('data/dataset_2_noisy.csv'):
        df2 = pd.read_csv('data/dataset_2_noisy.csv')
        datasets['DS2'] = {'name': 'Dataset 2 (10% šum, nepravilan)', 't': df2['t'].values, 'X': df2[['x', 'v']].values, 'x': df2['x'].values, 'noise': 'high', 'epochs': 5000}
    
    if os.path.exists('data/dataset_3_kaggle.csv'):
        df3 = pd.read_csv('data/dataset_3_kaggle.csv')
        df3.rename(columns={'time': 't', 'displacement': 'x', 'velocity': 'v'}, inplace=True)
        X3 = df3[['x', 'v']].values
            
        datasets['DS3'] = {
            'name': 'Dataset 3 (Kaggle podaci)', 
            't': df3['t'].values, 
            'X': X3, 
            'x': df3['x'].values, 
            'noise': 'unknown', 
            'epochs': 5000
        }
    else:
        print("⚠ UPOZORENJE: 'data/dataset_3_kaggle.csv' nije pronađen! DS3 će biti preskočen.")
    
    results = {}
    for key, ds in datasets.items():
        print(f"\n📊 Evaluacija: {ds['name']}")
        
        print("  ▶ Izvršavanje SINDy analize...")
        sindy_res = sindy_analyze(ds['t'], ds['X'], ds['name'], noise_level=ds['noise'])
        
        print(f"  ▶ Izvršavanje PINN-a ({ds['epochs']} epoha)...")
        pinn_res = pinn_analyze_wrapper(ds['t'], ds['x'], key, epochs=ds['epochs'])
        
        results[key] = {'sindy': sindy_res, 'pinn': pinn_res, 'dataset': ds}
        
    return results, datasets

# ---------------------------
# VIZUELIZACIJA
# ---------------------------
def plot_comprehensive_comparison(results, datasets):
    print("\n[Grafici] Generisanje i čuvanje komparativnih grafika...")
    if not results:
        print("Nema rezultata za plotovanje.")
        return

    dataset_keys = list(results.keys())
    labels = [k for k in dataset_keys]
    x_pos = np.arange(len(dataset_keys))
    width = 0.35
    
    # 1. GRAFIK: Poređenje MSE Greške (Linear i Log skala)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sindy_mse = [results[k]['sindy']['mse'] for k in dataset_keys]
    pinn_mse = [results[k]['pinn']['mse'] for k in dataset_keys]
    
    axes[0].bar(x_pos - width/2, sindy_mse, width, label='SINDy', color='#2196F3', alpha=0.8)
    axes[0].bar(x_pos + width/2, pinn_mse, width, label='PINN', color='#FF5722', alpha=0.8)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Poređenje MSE greške')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x_pos - width/2, sindy_mse, width, label='SINDy', color='#2196F3', alpha=0.8)
    axes[1].bar(x_pos + width/2, pinn_mse, width, label='PINN', color='#FF5722', alpha=0.8)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('MSE (Log skala)')
    axes[1].set_title('Poređenje MSE (Log skala)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.savefig('plots/comparison_mse.png')
    plt.close()

    # 2. GRAFIK: Odvojeni Bar Plot za koeficijent prigušenja (c)
    fig, ax = plt.subplots(figsize=(9, 6))
    sindy_c = [results[k]['sindy']['c'] for k in dataset_keys]
    pinn_c = [results[k]['pinn']['c'] for k in dataset_keys]
    ax.bar(x_pos - width/2, sindy_c, width, label='SINDy', color='#2196F3', alpha=0.8)
    ax.bar(x_pos + width/2, pinn_c, width, label='PINN', color='#FF5722', alpha=0.8)
    ax.axhline(y=C_TRUE, color='g', linestyle='--', linewidth=2.5, label=f'Teorijsko c = {C_TRUE} (DS1 & DS2)')
    ax.set_ylabel('Vrednost parametra c')
    ax.set_title('Identifikacija koeficijenta prigušenja c')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig('plots/comparison_bar_param_c.png')
    plt.close()

    # 3. GRAFIK: Odvojeni Bar Plot za krutost opruge (k)
    fig, ax = plt.subplots(figsize=(9, 6))
    sindy_k = [results[k]['sindy']['k'] for k in dataset_keys]
    pinn_k = [results[k]['pinn']['k'] for k in dataset_keys]
    ax.bar(x_pos - width/2, sindy_k, width, label='SINDy', color='#2196F3', alpha=0.8)
    ax.bar(x_pos + width/2, pinn_k, width, label='PINN', color='#FF5722', alpha=0.8)
    ax.axhline(y=K_TRUE, color='g', linestyle='--', linewidth=2.5, label=f'Teorijsko k = {K_TRUE} (DS1 & DS2)')
    ax.set_ylabel('Vrednost parametra k')
    ax.set_title('Identifikacija krutosti opruge k')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig('plots/comparison_bar_param_k.png')
    plt.close()

    # POJEDINAČNI GRAFICI ZA SVAKI DATASET (Trajektorije i Fazni prostori)
    for key in dataset_keys:
        ds = datasets[key]
        sr = results[key]['sindy']
        pr = results[key]['pinn']
        
        # Vremenska trajektorija x(t)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(ds['t'], ds['x'], color='gray', alpha=0.4, s=15, label='Podaci (Merenja)')
        if sr['x_sim_dense'] is not None:
            ax.plot(sr['t_dense'], sr['x_sim_dense'][:, 0], 'b--', lw=2, label=f"SINDy (MSE: {sr['mse']:.2e})")
        ax.plot(pr['t_dense'], pr['x_pred_dense'], 'r-', lw=2, label=f"PINN (MSE: {pr['mse']:.2e})")
        ax.set_title(f"Vremenska trajektorija x(t) - {ds['name']}")
        ax.set_xlabel('Vreme (t)')
        ax.set_ylabel('Pozicija (x)')
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.savefig(f"plots/comparison_{key.lower()}_trajectory.png")
        plt.close()

        # NOVO: Fazni prostor (Phase Space: x vs v)
        fig, ax = plt.subplots(figsize=(8, 6))
        # Izmereni podaci faznog prostora
        ax.plot(ds['X'][:, 0], ds['X'][:, 1], color='gray', alpha=0.5, lw=1.5, label='Podaci (Merenja)')
        
        # SINDy simulacija faznog prostora (x vs v)
        if sr['x_sim_dense'] is not None and sr['x_sim_dense'].shape[1] > 1:
            ax.plot(sr['x_sim_dense'][:, 0], sr['x_sim_dense'][:, 1], 'b--', lw=2, label='SINDy Rekonstrukcija')
            
        # PINN rekonstrukcija faznog prostora (x_pred vs v_pred)
        ax.plot(pr['x_pred_dense'], pr['v_pred_dense'], 'r-', lw=2, label='PINN Rekonstrukcija')
        
        ax.set_title(f"Fazni portret (Phase Space) - {ds['name']}")
        ax.set_xlabel('Pozicija (x)')
        ax.set_ylabel('Brzina (v)')
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.savefig(f"plots/comparison_{key.lower()}_phasespace.png")
        plt.close()

# ---------------------------
# ZAVRŠNI REZULTATI
# ---------------------------
def print_final_summary(results, datasets):
    print("\n" + "="*70)
    print("KONAČNI REZULTATI IDENTIFIKACIJE PARAMETARA")
    print("="*70)
    for key in results.keys():
        s = results[key]['sindy']
        p = results[key]['pinn']
        print(f"\n📊 {datasets[key]['name']}:")
        
        if key in ['DS1', 'DS2']:
            print(f"   Teorijski -> c: {C_TRUE:.4f} | k: {K_TRUE:.4f}")
        else:
            print(f"   (Teorijski parametri za Kaggle su nepoznati)")
            
        print(f"   SINDy     -> c: {s['c']:.4f} | k: {s['k']:.4f} | MSE: {s['mse']:.2e}")
        print(f"   PINN      -> c: {p['c']:.4f} | k: {p['k']:.4f} | MSE: {p['mse']:.2e}")

if __name__ == "__main__":
    print("🚀 Pokretanje kompletne analize sa pravim Kaggle podacima...")
    results, datasets = run_comparison()
    plot_comprehensive_comparison(results, datasets)
    print_final_summary(results, datasets)
    print("\n✅ Analiza uspješno završena! Svi grafici (uključujući fazne prostore i parametre c/k) su sačuvani u 'plots/'.")