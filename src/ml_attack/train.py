import torch
import torch.nn as nn
from .utils import get_vector_distribution

import torch.optim as optim
from .utils import clean_secret, check_secret

import numpy as np

# Neural solver model for Module-LWE using both Fourier mapping and FFT transformation.
class LinearComplex(nn.Module):
    def __init__(self, params):
        """
        n: Secret dimension (e.g., 8)
        q: Modulus
        """
        super(LinearComplex, self).__init__()
        self.q = params['q']
        self.n = params['n']
        self.k = params['k']
        self.secret_type = params['secret_type']
        self.params = params
        
        mean_s, _, std_s = get_vector_distribution(params, self.secret_type, params.get('hw'))

        self.guessed_secret = nn.Parameter(nn.init.normal_(torch.empty(self.n * self.k, dtype=torch.float), mean=mean_s, std=std_s), requires_grad=True)

    def forward(self, A_batch):
        return torch.tensordot(A_batch, self.guessed_secret, dims=1)
    

def train_until_stall(model, A_train, b_train, dataset, epoch=0, lr=1e-3, check_every=50, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    A_train = A_train.to(device)
    b_train = b_train.to(device)
    params = dataset.params

    model.train()

    loss_history = []
    lookback = 10
    min_decrease = -0.01

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)

    while True:
        optimizer.zero_grad()

        pred_b = model(A_train)
        b_loss = loss_fn(pred_b, b_train)
            
        b_loss.backward()
        optimizer.step()

        loss_history.append(b_loss.item())
        if verbose:
            print(f"Epoch {epoch}, Loss: {b_loss.item():.4f}")
        
        if epoch % check_every == 0:
            with torch.no_grad():
                guessed_secret = clean_secret(model.guessed_secret.cpu().detach().numpy(), params)
                if check_secret(guessed_secret, dataset.A, dataset.B, params):
                    return 0, epoch

        if len(loss_history) > lookback and \
            (np.mean([loss_history[i] - loss_history[i-1] 
                 for i in range(-lookback, 0)]) > min_decrease):
            return b_loss.item(), epoch
        
        epoch += 1


class TukeyLinearRegressor(nn.Module):
    def __init__(self, params, c=4.685, lr=1e-3, max_iter=1000, tol=1e-5, normalize=True):
        super().__init__()
        self.q = params['q']
        self.n = params['n']
        self.k = params['k']
        self.secret_type = params['secret_type']
        self.hw = params['hw']
        self.params = params

        # Training parameters
        self.c_orig = c  # original c value (user-defined)
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize

        # Guessed secret
        mean_s, _, std_s = get_vector_distribution(params, self.secret_type, self.hw)
        self.guessed_secret = nn.Parameter(
            torch.empty(self.n * self.k, dtype=torch.float32).normal_(mean=mean_s, std=std_s),
            requires_grad=True
        )

    def forward(self, X):
        return X @ self.guessed_secret

    def tukey_loss(self, residuals, c):
        abs_r = torch.abs(residuals)
        mask = abs_r <= c
        out = torch.zeros_like(residuals)
        r_c = residuals / c
        out[mask] = (c ** 2 / 6) * (1 - (1 - r_c[mask] ** 2) ** 3)
        out[~mask] = (c ** 2) / 6
        return out.mean()

    def fit(self, X_np, y_np):
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)

        if self.normalize:
            self.X_mean = X.mean(0)
            self.X_std = X.std(0)
            X = (X - self.X_mean) / self.X_std

            self.y_mean = y.mean()
            self.y_std = y.std()
            y = (y - self.y_mean) / self.y_std

            c_scaled = self.c_orig / self.y_std.item()
        else:
            c_scaled = self.c_orig

        optimizer = torch.optim.Adam([self.guessed_secret], lr=self.lr)

        for i in range(self.max_iter):
            optimizer.zero_grad()
            y_pred = self.forward(X)
            residuals = y - y_pred
            loss = self.tukey_loss(residuals, c_scaled)
            loss.backward()
            grad_norm = self.guessed_secret.grad.norm().item()
            optimizer.step()

            if grad_norm < self.tol:
                break

        with torch.no_grad():
            if self.normalize:
                scale = self.y_std / self.X_std
                self.coef_ = (self.guessed_secret * scale).detach().numpy()
                self.intercept_ = self.y_mean.item() - (self.coef_ * self.X_mean.numpy()).sum()
            else:
                self.coef_ = self.guessed_secret.detach().numpy()
                self.intercept_ = 0.0

        return self
    