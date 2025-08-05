from sympy import binomial, Integer
from scipy.stats import binom
from scipy.special import comb, erf, erfinv
import hashlib
import json
import numpy as np
from collections import defaultdict, Counter
import re
import ast
import math 

from typing import Optional, List, Dict, Any
import os 
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from statsmodels.robust.norms import TukeyBiweight, TrimmedMean
from statsmodels.robust.robust_linear_model import RLM
import statsmodels.api as sm

import time

import cpuinfo

_ALREADY_PATCHED = False

def patch_once():
    global _ALREADY_PATCHED
    if not _ALREADY_PATCHED and "intel" in cpuinfo.get_cpu_info().get("vendor_id_raw", "").lower():
        from sklearnex import patch_sklearn
        patch_sklearn()
        print("✔️ Patched scikit-learn (once).")
        _ALREADY_PATCHED = True


def get_slurm_cpu_count():
    return int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))

def increase_byte(input_bytes, N):
    return (int.from_bytes(input_bytes, byteorder='big') + N).to_bytes(len(input_bytes), byteorder='big')
        
def cmod(A, q):
    A %= q
    if isinstance(A, np.ndarray):
        A[A > q//2] -= q
    else:
        A = A - q if A > q//2 else A
    return A

def mod_mult(mat1, mat2, q):
    if np.log2(q) <= 32:
        return cmod(mat1 @ mat2, q)

    # Use 128-bit floats and scale the matrix slightly
    frac = 10_000
    mat1 = mat1.astype(np.float128)
    out = (mat1 // frac) @ (mat2 * frac % q)
    out += (mat1 % frac) @ mat2
    return cmod(out, q).astype(np.int64)

def time_execution(func):
    """
    Wrapper to calculate the time taken to execute a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def compute_b_candidates_and_probs(b_mod, mu, sigma, modulus, num_std, threshold):
    """
    For a batch of inputs, compute candidate b values and associated probabilities.
    Returns two lists of tensors: candidates and their probabilities.
    """
    min_vals = np.floor((mu - num_std * sigma) / modulus)
    max_vals = np.ceil((mu + num_std * sigma) / modulus)

    b_candidates_list = []
    b_probs_list = []

    for i in range(len(b_mod)):
        C_values = np.arange(min_vals[i], max_vals[i] + 1)
        candidates = C_values * modulus + b_mod[i]

        diffs = (candidates - mu[i]) / sigma[i]
        probs = np.exp(-0.5 * diffs ** 2)
        probs /= np.sum(probs)

        # Filtering
        mask = probs > threshold
        if np.sum(mask) > 0:
            candidates = candidates[mask]
            probs = probs[mask]
            probs /= np.sum(probs)

        b_candidates_list.append(candidates)
        b_probs_list.append(probs)

    return b_candidates_list, b_probs_list

def get_vector_distribution(params : dict, vector_type : str, hw : int = -1) -> tuple:
    """
    Returns the mean and variance of a vector based on its type.
    - params: parameters of the LWE scheme
    - vector_type: type of the vector (binary, ternary, cbd)
    - hw: hamming weight (optional, used for ternary and cbd)
    Returns:
    - mean: mean of the distribution
    - var: variance of the distribution
    - std: standard deviation of the distribution
    """
    n = params['n']

    match vector_type:
        case 'binary':
            if hw <= 0:
                mean = 0.5
                var = 0.25
            else:
                p = hw / n
                mean = p
                var = p * (1 - p)

        case 'ternary':
            mean = 0.0
            if hw <= 0:
                var = 2.0 / 3.0
            else:
                var = hw / n
        
        case 'gaussian':
            mean = 0.0
            var = params['gaussian_std'] ** 2
            if hw > 0:
                var *= hw / n

        case 'cbd':
            eta = params['eta']  # used only for CBD
            mean = 0.0
            if hw <= 0:
                var = eta / 2
            else:
                q = 1 - comb(2*eta, eta) * 2**(-2*eta)

                # precompute binomial pmf for M ~ Bin(n-1, q)
                m = np.arange(n)
                pmf = binom.pmf(m, n-1, q)

                alpha = pmf[:hw].sum() + np.sum(pmf[hw:] * (hw / (m[hw:]+1)))
                
                var_x = eta / 2

                var = var_x * alpha

        case _:
            raise ValueError("Unknown vector type")

    return mean, var, np.sqrt(var)

def get_b_distribution(params : dict, matrix, R = None) -> tuple:
    """
    Returns the approximate distribution of B values based on A and secret and error distributions.
    - params: parameters of the LWE scheme
    - matrix: either A or RA matrix
    - R: reduction matrix (optional)

    Returns:
    - mean_b: mean of the distribution
    - var_b: variance of the distribution
    - std_b: standard deviation of the distribution
    """
    if 'error_type' in params:
        mean_e, var_e, _ = get_vector_distribution(params, params['error_type'])
        if R is not None:
            mean_e = mean_e * np.sum(R, axis=-1)
            var_e = var_e * np.sum(np.pow(R, 2), axis=-1)
    else:
        mean_e, var_e = 0.0, 0.0

    matrix_sum = np.sum(matrix, axis=-1)
    matrix_sq_sum = np.sum(np.pow(matrix, 2), axis=-1)

    if 'hw' in params and params['hw'] > 0 and params['secret_type'] == 'binary':
        # In the binary case I have to take into account also the covariance matrix because 
        # the distribution does not have zero-mean (and so it is not simmetric)
        h = params['hw']
        n = params['n'] * params['k']
        mean_b = (h / n) * matrix_sum + mean_e 
        scaling = (h * (n - h)) / (n * (n-1))
        var_b = scaling * (matrix_sq_sum - ((matrix_sum**2) / n)) + var_e
    else:
        mean_s, var_s, _ = get_vector_distribution(params, params['secret_type'], params.get('hw', -1))
        mean_b = mean_s * matrix_sum + mean_e
        var_b = var_s * matrix_sq_sum + var_e

    return mean_b, var_b, np.sqrt(var_b)

def get_error_distribution(params : dict, R = None) -> tuple:
    """
    Returns the approximate distribution of B values based on A and secret and error distributions.
    - params: parameters of the LWE scheme
    - matrix: either A or RA matrix
    - R: reduction matrix (optional)

    Returns:
    - mean_e: mean of the distribution
    - var_e: variance of the distribution
    - std_e: standard deviation of the distribution
    """
    if 'error_type' in params:
        mean_e, var_e, _ = get_vector_distribution(params, params['error_type'])
        if R is not None:
            mean_e = mean_e * np.sum(R, axis=-1)
            var_e = var_e * np.sum(np.pow(R, 2), axis=-1)
    else:
        mean_e, var_e = 0.0, 0.0

    return mean_e, var_e, np.sqrt(var_e)

def get_percentage_true_b(dataset, top_percent=1, verbose=False, indices=None):
    exact_candidates = 0

    secret = dataset.get_secret()
    A_reduced = dataset.get_A()
    b_reduced = dataset.get_B()

    _, _, std_B = dataset.get_b_distribution()

    b_real = get_no_mod(dataset.params, A_reduced, secret, b_reduced)

    if indices is None:
        if top_percent < 1:
            # Select top N% with the lowest std
            num_selected = int(len(std_B) * top_percent)
            selected_indices = np.argsort(std_B)[:num_selected]
        else:
            selected_indices = np.arange(len(std_B))
    else:
        selected_indices = indices
    
    exact_candidates = np.sum(dataset.best_b[selected_indices] == b_real[selected_indices])
    total_selection = len(selected_indices)

    if verbose:
        if top_percent == 1:
            print(f"True B is the best candidate: {exact_candidates} / {total_selection} ({100 * exact_candidates / total_selection:.2f}%)")
        else:
            print(f"[BEST {int(top_percent*100)}% STD] True B is the best candidate: {exact_candidates} / {total_selection} ({100 * exact_candidates / total_selection:.2f}%)")

    return exact_candidates / total_selection

def get_expected_percentage_true_b(dataset, top_percent=1, verbose=False, indices=None):
    
    _, _, std_B = dataset.get_b_distribution()

    if indices is None:
        # Select top N% with the lowest std
        num_selected = int(len(std_B) * top_percent)
        selected_indices = np.argsort(std_B)[:num_selected]
    else:
        selected_indices = indices

    max_probs = [max(probs) for i, probs in enumerate(dataset.b_probs) if i in selected_indices] 
    expected_success_rate = np.mean(max_probs)
    if verbose:
        if top_percent == 1:
            print(f"Expected true B is best candidate: {100 * expected_success_rate:.2f}%")
        else:
            print(f"[BEST {int(top_percent*100)}% STD] Expected true B is best candidate: {100 * expected_success_rate:.2f}%")

    return expected_success_rate

def get_true_mask(dataset):
    """
    Returns a mask indicating which b values are true candidates.
    """
    secret = dataset.get_secret()
    A_reduced = dataset.get_A()
    b_reduced = dataset.get_B()

    b_real = get_no_mod(dataset.params, A_reduced, secret, b_reduced)

    mask = np.zeros(len(b_real), dtype=bool)
    for i in range(len(b_real)):
        if b_real[i] in dataset.b_candidates[i] and \
           b_real[i] == dataset.b_candidates[i][np.argmax(dataset.b_probs[i])]:
            mask[i] = True

    return mask

def compute_max_trials(confidence, p, min_samples, max_cap=10000):
    try:
        base_prob = p ** min_samples
        if base_prob >= 1:
            return 1
        denom = np.log(1 - base_prob)
        if denom == 0:
            return max_cap
        max_trials = np.ceil(np.log(1 - confidence) / denom)
        if np.isinf(max_trials) or np.isnan(max_trials):
            return max_cap
        return int(min(max_trials, max_cap))
    except:
        return max_cap
    
    
def train_model(dataset, A, b):
    if dataset.params['model'] == 'tukey':
        #model = TukeyRegressor(c=best_c,
        #                       lr=dataset.params['lr'], 
        #                       tol=dataset.params['tol'],
        #                       max_iter=dataset.params['max_iter'],
        #                       fit_intercept=dataset.params['fit_intercept'])
        if dataset.params['fit_intercept']:
            A = sm.add_constant(A)

        model = RLM(b, A, M=TukeyBiweight(c=dataset.params['c_factor']))

    elif dataset.params['model'] == 'huber':
        patch_once()

        # Scale
        scaler = StandardScaler()
        A_scaled = scaler.fit_transform(A)

        # Train the model using HuberRegressor
        model = HuberRegressor(max_iter= dataset.params['max_iter'],
                fit_intercept=dataset.params['fit_intercept'], 
                alpha=dataset.params['alpha'],
                tol=dataset.params['tol'],
                epsilon=dataset.params['epsilon'],
                warm_start=dataset.params['warm_start']
                )
    
    if dataset.params['use_ransac']:

        if dataset.params['residual_factor'] is not None:  
            std_B = dataset.get_error_distribution()[2]
            residual_threshold = dataset.params['residual_factor'] * np.max(std_B)
        else:
            residual_threshold = None
        
        _, _, std_B = dataset.get_b_distribution()
        p = np.mean(erf(dataset.mlwe.q / (2 * np.sqrt(2) * std_B)))

        if dataset.params['min_samples'] is None:
            min_samples = np.ceil((dataset.params['k'] * dataset.params['n'] + 1) / p).astype(int)
            min_samples = min(min_samples, len(A) - 1)
        else:
            min_samples = dataset.params['min_samples']

        confidence = 0.9
        optimal_max_trials = compute_max_trials(confidence, p, min_samples, dataset.params['max_trials'])

        if dataset.params['verbose']:
            print(f"Using RANSAC with min_samples={min_samples}, residual_threshold={residual_threshold}, max_trials={optimal_max_trials}")

        patch_once()
        model = RANSACRegressor(model, 
                                min_samples=min_samples, 
                                residual_threshold=residual_threshold,
                                max_trials=optimal_max_trials
                                )
        
        raw_secret_scaled = model.fit(A_scaled, b).estimator_.coef_
        raw_secret = raw_secret_scaled / scaler.scale_
    else:
        if dataset.params['model'] == 'tukey':
            #raw_secret = model.fit(A, b).coef_
            expected_s, _, std_s = get_vector_distribution(dataset.params, dataset.params['secret_type'], dataset.params['hw'])
            start_params = np.random.normal(loc=expected_s, scale=std_s, size=A.shape[1])
            results = model.fit(maxiter=dataset.params['max_iter'], 
                                tol=dataset.params['tol'], 
                                start_params=start_params,
                                conv='coefs')

            if dataset.params['fit_intercept']:
                raw_secret = results.params[1:]
            else:
                raw_secret = results.params

        elif dataset.params['model'] == 'huber':
            raw_secret_scaled = model.fit(A_scaled, b).coef_
            raw_secret = raw_secret_scaled / scaler.scale_

    guessed_secret = clean_secret(raw_secret, dataset.params)
    if check_secret(guessed_secret, dataset.A, dataset.B, dataset.params):
        return True, guessed_secret
    else:
        return False, guessed_secret

def report(real_secret, guessed_secret):
    """
    Print classification report and confusion matrix.
    """
    # Get unique sorted labels and compute confusion matrix
    labels = np.unique(np.concatenate((real_secret, guessed_secret)))
    cm = confusion_matrix(real_secret, guessed_secret, labels=labels)

    # Header
    header = "       |" + "".join([f"{l:>6}" for l in labels]) + " | Accuracy"
    print("Confusion Matrix:")
    print(header)
    print("-" * len(header))

    # Rows
    for i, row in enumerate(cm):
        label = f"{labels[i]:>6} |"
        values = "".join([f"{v:6}" for v in row])

        correct = row[i]
        total = row.sum()
        acc = correct / total if total > 0 else 0.0
        print(label + values + f" | {acc:4.1%}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(real_secret, guessed_secret, zero_division=0))


def check_secret(guessed_secret, A, B, params):
    """
    Checks if the secret is valid.
    """
    guessed_secret %= params['q']
    guessed_secret[guessed_secret > params['q'] // 2] -= params['q']
    
    e = B - np.tensordot(A, guessed_secret, axes=1)
    e %= params['q']

    e[e > params['q'] // 2] -= params['q']
    
    match params['error_type']:
        case 'binary':
            return np.allclose(e, np.zeros_like(e) + 0.5, atol=0.5)
        case 'ternary':
            return np.allclose(e, np.zeros_like(e), atol=1)
        case 'gaussian':
            expected_std = params['gaussian_std']
            actual_mean = np.mean(e)
            actual_std = np.std(e)

            mean_close = np.abs(actual_mean) < expected_std * 0.1
            std_close = np.abs(actual_std - expected_std) < expected_std * 0.2

            return mean_close and std_close
        case 'cbd':
            return np.allclose(e, np.zeros_like(e), atol=params['eta'])
        case _:
            raise ValueError("Unknown error type")


def clean_secret(secret, params: dict):
    """
    Correct the guessed secret to be in the right range and round it.
    """
    match params['secret_type']:
        case 'binary':
            secret = np.clip(np.round(secret), 0, 1)
        case 'ternary':
            secret = np.clip(np.round(secret), -1, 1)
        case 'cbd':
            secret = np.clip(np.round(secret), -params['eta'], params['eta'])
        case _:
            raise ValueError("Invalid secret type. Choose from 'binary', 'ternary', or 'cbd'.")
        
    secret[secret == -0.0] = 0.0
    return secret

def get_no_mod(params: dict, A, secret, B) -> bool:
    """
    Returns the no-mod condition for the given parameters and matrices.
    """
    q = params['q']

    secret = clean_secret(secret, params)
    A_s = A @ secret
    e = (B - A_s) % q
    e[e > q // 2] -= q

    real_b = A @ secret + e

    return real_b

def get_filename_from_params(params: dict, ext=".pkl"):
    """
    Generates a filename string based on the dataset parameters.

    Args:
        params (dict): Parameters of the dataset.
        prefix (str): Prefix for the filename.
        ext (str): File extension.

    Returns:
        str: Generated filename.
    """
    # Pick the important params you want in the filename
    key_mapping = {
        'n': 'n',
        'k': 'k',
        'secret_type': 's'
    }

    dir = params.get('save_to', './')
    if not dir.endswith('/'):
        dir += '/'

    parts = [dir + 'data']
    for key, name in key_mapping.items():
        if key in params:
            parts.append(f"{name}_{params[key]}")

    relevant_keys =[
        # LWE Scheme Parameters
        'n', 'q', 'k', 'secret_type', 'eta', 'gaussian_std',
        'hw', 'error_type', 'num_gen', 'add_noise', 'mod_q', 'seed',
        
        # Reduction Algorithm Parameters
        #'float_type', 'min_samples','algos', 'lookback',
        #'bkz_deltas', 'flatter_alphas', 'verbose', 'checkpoint_filename',
        #'reload_checkpoint', 'reduction_std', 
        'reduction_factor', 'reduction_resampling', 'penalty',
        'matrix_config', 'bkz_block_sizes'
        # Approximation Parameters
        #'approximation_std'
    ]

    hashable_params = {k: v for k, v in params.items() if k in relevant_keys}

    # Convert params to a JSON string with sorted keys for consistent hashing
    params_str = json.dumps(hashable_params, sort_keys=True)
    hash_digest = hashlib.sha256(params_str.encode()).hexdigest()[:5]

    # Append the hash to the filename and add the extension
    parts.append(hash_digest)
    filename = "_".join(parts) + ext
    return filename

def get_lwe_default_params(
            # Parameters for the LWE scheme
            n: int = 256,
            q: int = 3329,
            k: int = 1,
            secret_type: str = 'cbd',
            eta: int = 2,
            gaussian_std: float = 3,
            hw: int = -1,
            error_type: str = 'cbd',
            num_gen: int = 4,
            add_noise: bool = True,
            mod_q: bool = True,
            seed: Optional[int] = None,

            # Parameters for approximation
            approximation_std: int = 3,
            approximation_threshold: float = 0.01,

            # Parameters for the dataset
            save_to: str = './',
            ) -> Dict[str, Any]:
  return {
    # Parameters for the LWE scheme
    'n': n,
    'q': q,
    'k': k,
    'secret_type': secret_type,
    'eta': eta,
    'gaussian_std': gaussian_std,
    'hw': hw,
    'error_type': error_type,
    'num_gen': num_gen,
    'add_noise': add_noise,
    'mod_q': mod_q,
    'seed': seed,

    # Parameters for approximation
    'approximation_std': approximation_std,
    'approximation_threshold': approximation_threshold,
    
    # Parameters for the dataset
    'save_to': save_to
  }

def get_reduction_default_params(
            # Parameters for the reduction algorithm
            float_type: str = 'double',
            matrix_config: str = 'salsa',
            reduction_std: int = 2,
            reduction_factor: int = 1,
            reduction_resampling: bool = False,
            min_samples: int = 0,
            num_matrices: int = 0,
            algos: Optional[List[str]] = ['flatter', 'BKZ2.0'],
            lookback: Optional[List[int]] = 3,
            bkz_block_sizes: Optional[List[int]] = [30, 40],
            bkz_deltas: Optional[List[float]] = [0.96, 0.99],
            flatter_alphas: Optional[List[float]] = [0.04, 0.025],
            penalty: int = 4,
            verbose: bool = False,
            checkpoint_filename: str = './best_reduction',
            reload_checkpoint: bool = False,
            ) -> Dict[str, Any]:
  return {
    # Parameters for the reduction algorithm
    'float_type': float_type,
    'matrix_config': matrix_config,
    'reduction_std': reduction_std,
    'reduction_factor': reduction_factor,
    'reduction_resampling': reduction_resampling,
    'min_samples': min_samples,
    'num_matrices': num_matrices,
    'algos': algos,
    'lookback': lookback,
    'bkz_block_sizes': bkz_block_sizes,
    'bkz_deltas': bkz_deltas,
    'flatter_alphas': flatter_alphas,
    'penalty': penalty,
    'verbose': verbose,
    'checkpoint_filename': checkpoint_filename,
    'reload_checkpoint': reload_checkpoint
  }

def get_continuous_reduction_default_params(
            # Parameters for the reduction algorithm
            float_type: str = 'd',
            matrix_config: str = 'dual',
            reduction_samples: Optional[int] = None,
            reduction_resampling: bool = True,
            min_samples: int = 0,
            num_matrices: int = 0,
            reduction_max_size: int = 1000,
            lookback: int = 3,
            warmup_steps: int = 10,
            flatter_alpha: float = 0.001,
            bkz_delta: float = 0.99,
            bkz_block_sizes: str = '20:50:5',
            #crossover: int = -1,
            interleaved_steps: int = 0,
            use_polish: bool = True,
            penalty: int = 4,
            verbose: bool = False
            ) -> Dict[str, Any]:
  return {
    # Parameters for the reduction algorithm
    'float_type': float_type,
    'matrix_config': matrix_config,
    'reduction_samples': reduction_samples,
    'reduction_resampling': reduction_resampling,
    'min_samples': min_samples,
    'num_matrices': num_matrices,
    'reduction_max_size': reduction_max_size,
    'lookback': lookback,
    'warmup_steps': warmup_steps,
    'flatter_alpha': flatter_alpha,
    'bkz_delta': bkz_delta,
    'bkz_block_sizes': bkz_block_sizes,
    'use_polish': use_polish,
    #'crossover': crossover,
    'interleaved_steps': interleaved_steps,
    'penalty': penalty,
    'verbose': verbose
  }

def get_train_default_params(
        train_percentages: List[float] = [1.0],
        model: str = 'tukey',
        lr: float = 0.0001,
        c_factor: float = 1.0,
        epsilon: float = 1.1,
        max_iter: int = 15000,
        alpha: float = 0.0001,
        warm_start: bool = False,
        fit_intercept: bool = False,
        tol: float = 0.0001,
        use_ransac: bool = False,
        residual_factor: Optional[float] = 1.5,
        min_samples: Optional[int] = None,
        max_trials: int = 100
        ):
    return {
        "train_percentages": train_percentages,
        "model": model,
        "lr": lr,
        "c_factor": c_factor,
        "epsilon": epsilon,
        "max_iter": max_iter,
        "alpha": alpha,
        "warm_start": warm_start,
        "fit_intercept": fit_intercept,
        "tol": tol,
        "use_ransac": use_ransac,
        "residual_factor": residual_factor,
        "min_samples": min_samples,
        "max_trials": max_trials
    }

def get_default_params():
    params = get_lwe_default_params()
    params.update(get_continuous_reduction_default_params())
    params.update(get_train_default_params())
    return params

def prob_all_seen(n, m, k):
    total = Integer(0)
    sign = -1
    for j in range(n+1):
        if n - j < m:
            break
        
        sign = -1 if sign == 1 else 1
        term = sign * binomial(n, j) * binomial(n - j, m)**k
        total += term
    
    return float(total / binomial(n, m)**k)

def calculate_min_trials(n, m, target_prob=0.99, max_k=1000):
    """
    Finds minimal k such that P(all n items seen) ≥ target_prob.
    Uses binary search for efficiency.
    """
    if m >= n:
        return 1
    
    low = 1
    high = max_k
    answer = high
    
    while low <= high:
        mid = (low + high) // 2
        prob = prob_all_seen(n, m, mid)
        
        if prob >= target_prob:
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    
    return answer

def polish(X, longtype=False):
    if longtype:
        X = X.astype(np.longdouble)
    g, old = np.inner(X, X), np.inf  # Initialize the Gram matrix
    while np.std(X) < old:
        old = np.std(X)
        # Calculate the projection coefficients
        c = np.round(g / np.diag(g)).T.astype(int)
        c[np.diag_indices(len(X))] = 0
        c[np.diag(g) == 0] = 0

        sq = np.diag(g) + c * ((c.T * np.diag(g)).T - 2 * g)  # doing l2 norm here
        s = np.sum(sq, axis=1)  # Sum the squares. Can do other powers of norms
        it = np.argmin(s)  # Determine which index minimizes the sum
        X -= np.outer(c[it], X[it])  # Project off the it-th vector
        g += np.outer(c[it], g[it][it] * c[it] - g[it]) - np.outer(
            g[:, it], c[it]
        )  # Update the Gram matrix
    return X

def parse_range(range_str):
    start, stop, step = map(int, range_str.split(":"))
    list_items = list(range(start, stop, step))
    if stop not in list_items:
        list_items.append(stop)
    return list_items

def get_hermite_root_factor(b):
    """
    Computes the Hermite root factor for a given block size.
    The formula is derived from the approximation of the Hermite constant.
    """
    return ((b / (2 * np.pi * np.e)) * ((b * np.pi ) ** (1 / b))) ** (1/(2*(b-1)))

def get_optimal_sample_size(nk, q, w, delta_0):
    """ Computes the optimal sample size m. """
    return int(np.sqrt(nk* (np.log(q) - np.log(w)) / np.log(delta_0)) - nk)

def get_optimal_vector_norm(n, m, q, w, b):
    """ Computes the optimal vector norm v. """
    delta_0 = get_hermite_root_factor(b)
    d = n + m
    return (delta_0 ** d) * np.exp((n * np.log(q) + m * np.log(w)) / d)

def prob_to_std(p: float, q: float) -> float:
    """
    Compute the minimum standard deviation of LWE noise (e)
    such that Pr[|e| < q/2] = p, where e ~ N(0, sigma^2).
    
    Parameters:
    - p: desired success probability (0 < p < 1)
    - q: modulus (positive number)
    
    Returns:
    - sigma: minimum standard deviation
    """
    if not (0 < p < 1):
        raise ValueError("Probability p must be between 0 and 1 (exclusive).")
    if q <= 0:
        raise ValueError("Modulus q must be positive.")

    sigma = q / (2 * math.sqrt(2) * erfinv(p))
    return sigma

def std_to_prob(sigma: float, q: float) -> float:
    if sigma <= 0:
        raise ValueError("Standard deviation sigma must be positive.")
    if q <= 0:
        raise ValueError("Modulus q must be positive.")

    p = erf(q / (2 * math.sqrt(2) * sigma))
    return p

def cbd_expected_hamming_weight(n, eta):
    """
    Fast approximation of expected Hamming weight for CBD_eta samples.
    """
    p_zero = 1 / np.sqrt(np.pi * eta)
    p_non_zero = 1 - p_zero
    return int(n * p_non_zero)

def pad_vectors_to_max(vectors_list):
    max_len = max(v.shape[0] for v in vectors_list)
    min_len = min(v.shape[0] for v in vectors_list)

    if max_len == min_len:
        return np.stack(vectors_list).astype(int)
    
    padded = []
    for v in vectors_list:
        pad_size = max_len - v.shape[0]
        if pad_size > 0:
            padding = np.zeros((pad_size, v.shape[1]), dtype=int)
            v_padded = np.vstack((v, padding)).astype(int)
        else:
            v_padded = v
        padded.append(v_padded)

    return np.stack(padded).astype(int)

def parse_output_file(output_file):
    """Parse the output file and extract relevant statistics."""
    print(f"Parsing output file: {output_file}")

    # Statistics from the output file
    with open(output_file, 'r') as file:
        content = file.readlines()
        processed_content = []
        for line in content:
            if '.-' in line:
                split_lines = line.replace('.-', '.\n-').splitlines()
                processed_content.extend(split_lines)
            else:
                processed_content.append(line)
        content = [line for line in processed_content if line.strip()]

    # Prepare lists to collect the results
    results = {
        "output_file": os.path.basename(output_file),
        "exp_perc": Counter(),
        "best_b": Counter(),
        "exp_perc_subset": defaultdict(dict),
        "best_b_subset": defaultdict(dict),
        "tour_to_time": defaultdict(),
        "block_sizes": defaultdict(list),
        "num_updates": defaultdict(list),
        "std_B": defaultdict(float),
        "start_bkz": None,
        "total_time": 0,
        "total_train_samples": 0,
        "confusion_matrix": None,
        "classification_report": None,
        "success": False,
        "parameters": {},
        "num_matrices": None,
        "saved_to": None
    }

    # Extract dictionary from the first line using ast.literal_eval for safety
    parameters_line = content[0]
    results['parameters'] = ast.literal_eval(parameters_line.strip().split("Parameters: ")[1])

    # Extract number of matrices from the line "Attacking X matrices"
    matrices_line = content[1]
    num_matrices_match = re.search(r"Attacking (\d+) matrices*", matrices_line)
    results['num_matrices'] = int(num_matrices_match.group(1)) if num_matrices_match else None

    # Iterate through the file content and parse relevant data
    counter = 2
    current_tour = 0
    while counter < len(content):
        line = content[counter]
        if line.startswith("Tour"):
            # Extract tour number
            match = re.search(r'Tour (\d+)', line)
            current_tour = int(match.group(1)) - 1 if match else current_tour + 1

            # Extract time from "Tour X completed after Y seconds."
            time_match = re.search(r'Tour \d+ completed after ([\d.]+) seconds', line)
            if time_match:
                tour_time = float(time_match.group(1)) / 3600 # Convert seconds to hours
                results['tour_to_time'][current_tour] = tour_time

            # Case 2: "Time" only (e.g., "Time: 123.45s")
            match_time_only = re.search(r'Time: ([\d.]+)s', line)
            if match_time_only:
                tour_time = float(match_time_only.group(1)) / 3600  # Convert seconds to hours
                results['tour_to_time'][current_tour] = tour_time

            # Case 3: "Mean std_B" only (e.g., "Mean std_B: 456.78")
            match_std_only = re.search(r"Mean std_B: ([\d.]+)", line)
            if match_std_only:
                results['std_B'][current_tour] = float(match_std_only.group(1))

            # Start scanning lines until next block or BKZ
            counter += 1
            while counter < len(content):
                subline = content[counter]

                # Match [BEST X% STD] True B in candidate set
                match_true_b = re.match(r"\[BEST (\d+)% STD\] Expected true B is best candidate: ([\d.]+)%", subline)
                if match_true_b:
                    percentile = int(match_true_b.group(1))
                    percent = float(match_true_b.group(2))
                    results['exp_perc_subset'][current_tour][percentile] = percent
                    counter += 1
                    continue

                # Match [BEST X% STD] True B is the best candidate
                match_best = re.match(r"\[BEST (\d+)% STD\] True B is the best candidate: \d+ / \d+ \(([\d.]+)%\)", subline)
                if match_best:
                    percentile = int(match_best.group(1))
                    percent = float(match_best.group(2))
                    results['best_b_subset'][current_tour][percentile] = percent
                    counter += 1
                    continue

                # Match general True B
                match_general_true_b = re.match(r"Expected true B is best candidate: ([\d.]+)%", subline)
                if match_general_true_b:
                    results['exp_perc'][current_tour] = float(match_general_true_b.group(1))
                    counter += 1
                    continue

                # Match general Best candidate
                match_general_best = re.match(r"True B is the best candidate: \d+ / (\d+) \(([\d.]+)%\)", subline)
                if match_general_best:
                    results['total_train_samples'] = int(match_general_best.group(1))
                    results['best_b'][current_tour] = float(match_general_best.group(2))
                    counter += 1
                    continue

                # Match [BEST X% STD] True B in candidate set: N / M (P%)
                match_in_candidate = re.match(r"\[BEST (\d+)% STD\] True B in candidate set: (\d+) / (\d+) \(([\d.]+)%\)", subline)
                if match_in_candidate:
                    counter += 1
                    continue

                match_in_candidate = re.match(r"True B in candidate set: (\d+) / (\d+) \(([\d.]+)%\)", subline)
                if match_in_candidate:
                    counter += 1
                    continue

                # Match "Mean overall std_B: <float>"
                match_mean_std_b = re.match(r"Mean overall std_B: ([\d.]+)", subline)
                if match_mean_std_b:
                    results['std_B'][current_tour] = float(match_mean_std_b.group(1))
                    counter += 1
                    continue

                # If no match, break to the next tour
                counter -= 1
                break
        elif line.startswith("- Running BKZ2.0"):
            if results['start_bkz'] is None:
                results['start_bkz'] = current_tour
            block_size_match = re.search(r"with block size (\d+)...", line)
            if block_size_match:
                block_size = int(block_size_match.group(1))
                results['block_sizes'][current_tour].append(block_size)
        elif line.startswith("- Updated"):
            num_updates_match = re.search(r"Updated (\d+)/", line)
            if num_updates_match:
                results['num_updates'][current_tour].append(int(num_updates_match.group(1)))
        elif line.startswith("Reduction completed "):
            total_time_match = re.search(r"Reduction completed in (\d+\.\d+) seconds", line)
            if total_time_match:
                results['total_time'] += float(total_time_match.group(1))
        elif line.startswith("Secret found after"):
            total_time_match = re.search(r"Secret found after (\d+\.\d+) seconds.", line)
            if total_time_match:
                results['total_time'] += float(total_time_match.group(1))
        elif line.startswith("Confusion Matrix"):
            # Save lines for confusion matrix until classification report
            confusion_matrix_lines = []
            counter += 1
            while counter < len(content) and not content[counter].startswith("Classification Report"):
                confusion_matrix_lines.append(content[counter])
                counter += 1
            results['confusion_matrix'] = "".join(confusion_matrix_lines)
            counter -= 1
        elif line.startswith("Classification Report"):
            classification_report_lines = []
            counter += 1
            while counter < len(content) and not content[counter].startswith("#########"):
                classification_report_lines.append(content[counter])
                if "weighted avg" in content[counter]:
                    break
                counter += 1
            results['classification_report'] = "".join(classification_report_lines)
        elif line.startswith("Mean overall std_B:"):
            # Extract mean overall std_B
            match = re.search(r"Mean overall std_B: ([\d.]+)", line)
            if match:
                results['std_B'][current_tour] = float(match.group(1))
        elif line.startswith("Dataset saved to"):
            match = re.search(r"Dataset saved to (.+)", line)
            if match:
                results['saved_to'] = match.group(1)
        elif line.startswith("#########################################"):
            # End of attack
            break
        counter += 1

    results['success'] = True if results['confusion_matrix'] is not None else False

    return results

def extract_filenumber(f):
    return float(os.path.splitext(os.path.basename(f))[0].split('_')[-1])