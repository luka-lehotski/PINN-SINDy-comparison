"""
pinn_model.py
PINN (Physics-Informed Neural Network) za identifikaciju parametara
prigušenog harmonijskog oscilatora.

ROBUSNA VERZIJA ZA NEPRAVILNO UZORKOVANJE (DS2):
  - Gubitak podataka se računa na nepravilnim tačkama uzorka.
  - Fizika (loss_physics) se evaluira ISKLJUČIVO na čistom, uniformnom gridu (linspace)
    čime se eliminiše uticaj nepravilnog uzorkovanja na stabilnost Autograda.
"""

import torch
import torch.nn as nn
import numpy as np
import random

# DODAJ OVO ZA POTPUNU REPRODUKTIVNOST
def set_seed(seed=42):
    """Fiksira svu nasumičnost kako bi PINN uvek davao identičan rezultat."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Odmah pozovi funkciju sa nekim brojem (npr. 42, 100, 1234...)
set_seed(1234)

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i+1])
            if i < len(layers) - 2:
                nn.init.xavier_normal_(linear.weight)
                nn.init.zeros_(linear.bias)
            else:
                nn.init.xavier_uniform_(linear.weight, gain=0.1)
                nn.init.zeros_(linear.bias)
            self.net.append(linear)
        self.activation = nn.Tanh()

    def forward(self, tau):
        x = tau
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net) - 1:
                x = self.activation(x)
        return x


class DampedPINN:
    def __init__(self, t_data, x_data, epochs=5000, is_noisy=False):
        # Podaci mogu biti nepravilno uzorkovani - čuvamo ih kakvi jesu za Data Loss
        self.t_raw  = torch.tensor(t_data, dtype=torch.float64).reshape(-1, 1)
        self.x_data = torch.tensor(x_data, dtype=torch.float64).reshape(-1, 1)

        self.t_min = float(self.t_raw.min())
        self.t_max = float(self.t_raw.max())
        self.t_data = self._norm(self.t_raw)

        self.x_scale = float(self.x_data.abs().max()) + 1e-8
        self.x_data_norm = self.x_data / self.x_scale

        # Stabilna arhitektura
        self.model = PINN([1, 64, 64, 64, 1]).double()
        
        # Inicijalizacija parametara podalje od tačnih rješenja
        self.c = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.k = nn.Parameter(torch.tensor([3.0], dtype=torch.float64))

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': 1e-3},
            {'params': [self.c, self.k],         'lr': 1e-2},
        ])
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=250, factor=0.5, min_lr=1e-5
        )
        self.epochs = epochs
        self.is_noisy = is_noisy
        
        self.loss_history = []
        self.c_history = []
        self.k_history = []

    def _norm(self, t):
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

    def net_f(self, tau):
        """ Računanje izvoda diferencijalne jednačine """
        scale = 2.0 / (self.t_max - self.t_min)
        tau = tau.clone().requires_grad_(True)

        x_norm = self.model(tau)
        x = x_norm * self.x_scale

        x_tau = torch.autograd.grad(
            x, tau, torch.ones_like(x), create_graph=True, retain_graph=True
        )[0]
        x_tau_tau = torch.autograd.grad(
            x_tau, tau, torch.ones_like(x_tau), create_graph=True, retain_graph=True
        )[0]

        x_t  = x_tau     * scale
        x_tt = x_tau_tau * (scale ** 2)
        return x_tt + self.c * x_t + self.k * x

    def train(self, verbose=True):
        # Za šumne i nepravilne podatke (DS2), fizika odmah startuje snažno (0.5) 
        # kako bi delovala kao glatki filter nad nepravilnim podacima
        lambda_start = 0.50 if self.is_noisy else 0.05
        lambda_end   = 1.0

        if verbose:
            print(f"    [PINN] Trening otpočeo. Režim za nepravilan uzorak (DS2) = {self.is_noisy}")

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            progress = epoch / self.epochs
            lam = lambda_start + (lambda_end - lambda_start) * progress

            # 1. DATA LOSS: Računamo isključivo na stvarnim, nepravilnim tačkama podataka
            x_pred_norm = self.model(self.t_data)
            loss_data = torch.mean((x_pred_norm - self.x_data_norm) ** 2)

            # 2. PHYSICS LOSS: Generišemo savršeno čist, uniforman grid nezavisan od podataka!
            # Ovo sprečava anomalije u izvodima koje izaziva nepravilan vremenski korak.
            tau_linspace = torch.linspace(-1.0, 1.0, 300, dtype=torch.float64).reshape(-1, 1)
            
            f = self.net_f(tau_linspace)
            loss_physics = torch.mean(f ** 2) / (self.x_scale ** 2)

            # Ukupan gubitak
            loss = loss_data + lam * loss_physics
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

            self.loss_history.append(loss.item())
            self.c_history.append(self.c.item())
            self.k_history.append(self.k.item())

            if verbose and ((epoch + 1) % 500 == 0 or epoch == 0):
                print(f"      Epoha {epoch+1:4d}/{self.epochs} | Loss: {loss.item():.5f} | c: {self.c.item():.4f}, k: {self.k.item():.4f}")

    def predict(self, t):
        t_tensor = torch.tensor(t, dtype=torch.float64).reshape(-1, 1)
        tau = self._norm(t_tensor)
        self.model.eval()
        with torch.no_grad():
            return (self.model(tau) * self.x_scale).numpy().flatten()