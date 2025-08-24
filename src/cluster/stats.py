import numpy as np
import pickle
from pathlib import Path

import sys, os, argparse, json

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from kyber.mlwe import MLWE
from ml_attack.utils import (
    get_default_params, mod_mult, get_b_distribution, cmod, get_no_mod,
    compute_b_candidates_and_probs
)
from ml_attack.lwe import transform_vector_lwe

def get_data_salsa(data_path, updated_params, num_secrets, top_percent=1.0):
    """
    Loads the dataset from a Salsa directory with:
    - params.pkl: parameters of the dataset
    - origA_n.._logq...npy: original A matrix
    - data.prefix: reduced matrices
    """
    data_path = Path(data_path)

    with open(data_path / "params.pkl", 'rb') as f:
        loaded_params = pickle.load(f)
    
    params = get_default_params()
    
    params['k'] = loaded_params['rlwe']
    params['n'] = loaded_params['N'] // loaded_params['rlwe']
    params['q'] = loaded_params['Q']
    params['seed'] = loaded_params['seed']
    params['secret_type'] = 'cbd' if loaded_params['secret_type'] == 'binomial' else loaded_params['secret_type']
    params['eta'] = loaded_params['gamma']
    params['gaussian_std'] = loaded_params['sigma']
    params['hw'] = loaded_params['max_hamming']
    params['error_type'] = 'cbd' if loaded_params['secret_type'] == 'binomial' else 'gaussian'
    params['reduction_samples'] = loaded_params['m'] if loaded_params['m'] > 0 else 1
    m = loaded_params['m'] if loaded_params['m'] > 0 else loaded_params['N']
    params['reduction_resampling'] = True

    if params['k'] != 1:
        orig_A_path = data_path / f"origA_n{params['n']}_k{params['k']}_logq{int(np.ceil(np.log2(params['q'])))}.npy"
    else:
        orig_A_path = data_path / f"origA_n{params['n']}_logq{int(np.ceil(np.log2(params['q'])))}.npy"

    A = np.load(orig_A_path)
    params['num_gen'] = A.shape[0] // (params['n'] * params['k'])

    params.update(updated_params)

    print("Params:", params)

    full_R = []
    full_indices = []
    with open(data_path / "data.prefix") as fd:
        indices, RT = [], []
        for line in fd:
            if not line:
                continue
            ind, r = line.strip().split(";")
            indices.append(int(ind.strip()))
            RT.append(np.array(r.split(), dtype=np.int64))
            if len(indices) == m:
                R = np.array(RT).T
                if top_percent < 1.0:
                    # Select only the top percent of the rows
                    num_rows = int(len(R) * top_percent)
                    RA = mod_mult(R, A[indices], mlwe.q)
                    non_zero_indices = np.any(RA != 0, axis=-1)
                    _, _, std_B = get_b_distribution(params, RA[non_zero_indices], R[non_zero_indices])
                    sorted_indices = np.argsort(std_B)[:num_rows]
                    R = R[sorted_indices]
                full_indices.append(indices)
                full_R.append(R)
                indices, RT = [], []
        
    R = np.stack(full_R)
    indices = np.stack(full_indices)
    return {"R": R, "indices": indices, "params": params, "A": A}

def get_data_reduced(filepath):
    """
    Loads the dataset from a file without using the `MLWE` class.
    Assumes `indices` are not None and avoids computing `RA` and `RB`.
    """
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)

    params = loaded_data['params']
    A = loaded_data['A']

    if 'RC' in loaded_data:
        RC = loaded_data['RC']
        try:
            best_RC = loaded_data['best_RC']

            if params['matrix_config'] in ['salsa', 'dual']:
                m = A.shape[1]
                R = np.stack([reduced_matrix[:, :m] / loaded_data['params']['penalty'] for reduced_matrix in best_RC])
            else:
                n = A.shape[2]
                R = np.stack([reduced_matrix[:, n:] / loaded_data['params']['penalty'] for reduced_matrix in best_RC])

            if params['k'] == 1 and params['reduction_samples'] == 1 and not params['reduction_resampling']:
                R = np.stack([np.stack([neg_circ(row).T for row in reduced_matrix]) for reduced_matrix in R])
        except:
            print("Warning: 'best_RC' corrupted. Using 'RC' instead.")
            best_RC = None
            if params['matrix_config'] in ['salsa', 'dual']:
                m = A.shape[1]
                R = np.stack([reduced_matrix[:, :m] / loaded_data['params']['penalty'] for reduced_matrix in RC])
            else:
                n = A.shape[2]
                R = np.stack([reduced_matrix[:, n:] / loaded_data['params']['penalty'] for reduced_matrix in RC])
    else:
        R = loaded_data['R']

    indices = loaded_data['indices']  # Assumes indices are always present

    print("Params:", params)

    return {"R": R, "indices": indices, "params": params, "A": A}

def statistics(data):
    R = data["R"]
    indices = data["indices"]
    params = data["params"]
    A = data["A"]

    is_rlwe = params['k'] == 1 and params['reduction_samples'] == 1 and not params['reduction_resampling']

    A_to_reduce = np.stack([A[ind] for ind in indices])
    if is_rlwe:
        A_to_reduce = A_to_reduce[:, np.newaxis, :, :]

    RA = mod_mult(R, A_to_reduce, params['q'])
    non_zero_indices = np.any(RA != 0, axis=-1)

    std_A = np.mean(np.std(A_to_reduce, axis=-1)).astype(np.float64)
    std_RA = np.mean(np.std(RA[non_zero_indices], axis=-1)).astype(np.float64)
    reduction_factor = std_RA / std_A if std_A != 0 else 0

    print("Statistics A:")
    print(f" - Rho: {reduction_factor}")
    print(f" - Std A: {std_A}")
    print(f" - Std RA: {std_RA}")

    expected_b, var_b, std_b = get_b_distribution(params, RA[non_zero_indices], R[non_zero_indices])

    print("Approximation B:")
    print(f" - Expected B: {expected_b}")
    print(f" - Var B: {var_b}")
    print(f" - Std B: {std_b}")

    mlwe = MLWE(params)
    total_matches_nomod = []
    total_matches_approx = []
    for count in range(num_secrets):
        print(f"Generating secret {count + 1}/{num_secrets}")

        random_bytes = mlwe.get_random_bytes()
        secret_mlwe = mlwe.generate_secret(random_bytes)
        secret = transform_vector_lwe(secret_mlwe.to_list())
        secret = cmod(secret, mlwe.q)

        A_split = np.array_split(A, params['num_gen'])
        B = np.zeros((params['num_gen'] * params['n'] * params['k']))

        for i in range(params['num_gen']):
            random_bytes = mlwe.get_random_bytes()
            B[i * params['n'] * params['k']:(i + 1) * params['n'] * params['k']] = cmod(
                mod_mult(A_split[i], secret, mlwe.q) + transform_vector_lwe(mlwe.generate_error(random_bytes).to_list()), mlwe.q
            )

        B_to_reduce = np.stack([B[ind] for ind in indices])
        if is_rlwe:
            B_to_reduce = B_to_reduce[:, np.newaxis, :]

        RB = np.squeeze(mod_mult(R, B_to_reduce[..., np.newaxis], mlwe.q), axis=-1)

        b_real = get_no_mod(RA[non_zero_indices], secret, RB[non_zero_indices], mlwe.q)

        print("  Real B:")
        print(f"  - Expected B: {np.mean(b_real)}")
        print(f"  - Var B: {np.var(b_real)}")
        print(f"  - Std B: {np.std(b_real)}")

        matches_nomod = np.sum(RB[non_zero_indices] == b_real)
        print(f"  - Matches (NoMod): {matches_nomod} out of {len(b_real)} ({(matches_nomod / len(b_real)) * 100:.2f}%)")
        total_matches_nomod.append(matches_nomod)

        # Perform mod matching
        mod_matches = np.sum((RB[non_zero_indices] % mlwe.q) == (b_real))
        print(f"  - Mod Matches: {mod_matches} out of {len(b_real)} ({(mod_matches / len(b_real)) * 100:.2f}%)")

        # Approximate to get the best b and check again the match
        b_candidates, b_probs = compute_b_candidates_and_probs(
            b_mod=RB[non_zero_indices] % mlwe.q,
            mu=expected_b,
            sigma=std_b,
            modulus=mlwe.q,
            num_std=params['approximation_std'],
            threshold=params['approximation_threshold']
        )

        best_b = np.array([b_candidates[i][np.argmax(probs)] for i, probs in enumerate(b_probs)])

        print("  Approximation B (after refinement):")
        print(f"  - Best B: {np.mean(best_b)}")
        print(f"  - Var Best B: {np.var(best_b)}")
        print(f"  - Std Best B: {np.std(best_b)}")

        matches_approx = np.sum(best_b == b_real)
        print(f"  - Matches (refined): {matches_approx} out of {len(b_real)} ({(matches_approx / len(b_real)) * 100:.2f}%)")
        total_matches_approx.append(matches_approx)

    mean_matches_nomod = np.mean(total_matches_nomod)
    mean_matches_approx = np.mean(total_matches_approx)
    print(f"\nMean Matches (NoMod): {mean_matches_nomod}")
    print(f"Mean Matches (Approx): {mean_matches_approx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check NoMod Salsa")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the Salsa dataset directory")
    parser.add_argument("--params", type=str, required=True, help="Path to the JSON file with updated parameters")
    parser.add_argument("--top_percent", type=float, default=1.0, help="Top percentage of rows to select")
    parser.add_argument("--num_secrets", type=int, default=1, help="Number of secrets to generate and test")
    args = parser.parse_args()

    # Load updated parameters from the JSON file
    with open(args.params, 'r') as f:
        updated_params = json.load(f)

    if Path(args.data_path).is_dir():
        data = get_data_salsa(args.data_path, updated_params, args.num_secrets, args.top_percent)
    else:
        data = get_data_reduced(args.data_path)

    statistics(data)