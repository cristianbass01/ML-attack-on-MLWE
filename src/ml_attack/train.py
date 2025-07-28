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
