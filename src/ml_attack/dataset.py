from kyber.mlwe import MLWE

from ml_attack.lwe import transform_matrix_lwe, transform_vector_lwe
from ml_attack.reduction import Reduction
from ml_attack.continuous_reduction import ContinuousReduction
from ml_attack.utils import (
  get_b_distribution, 
  compute_b_candidates_and_probs, 
  calculate_min_trials, 
  get_default_params, 
  get_slurm_cpu_count,
  train_model,
  report,
  get_filename_from_params,
  get_no_mod,
  cmod,
  parse_range,
  get_hermite_root_factor,
  get_optimal_sample_size,
  get_optimal_vector_norm,
  pad_vectors_to_max,
  get_error_distribution,
  std_to_prob
)
from ml_attack.lwe import neg_circ

import numpy as np
import time

import pickle

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

class LWEDataset():
    def __init__(self, params: dict):
        """
        params for the LWE scheme
        """
        self.params = get_default_params()
        self.params.update(params)
        self.mlwe = MLWE(self.params)

        self.secret = None
        self.A = None
        self.B = None
        self.R = None
        self.RC = None
        self.best_RC = None
        self.RA = None
        self.RB = None
        self.reduced = False
        self.indices = None
        self.non_zero_indices = None
        self.reduction_time = 0
        
    def initialize(self):
        """
        Initializes the dataset by generating the LWE samples.
        """
        n = self.params['n']
        k = self.params['k']
        num_gen = self.params['num_gen']
        add_noise = self.params['add_noise']
        mod_q = self.params['mod_q']

        # Initialize A and B matrices
        self.A = np.zeros((num_gen *n*k, k*n))
        self.B = np.zeros((num_gen *n*k))
        
        random_bytes = self.mlwe.get_random_bytes()
        secret_mlwe = self.mlwe.generate_secret(random_bytes)
        self.secret = transform_vector_lwe(secret_mlwe.to_list())
        self.secret = cmod(self.secret, self.mlwe.q)

        for i in range(num_gen):
            random_bytes = self.mlwe.get_random_bytes()
            
            if add_noise:
                if mod_q:
                    A, B = self.mlwe.generate_A_B(secret_mlwe, random_bytes)
                    A_lwe = transform_matrix_lwe(A.to_list())
                    B_lwe = transform_vector_lwe(B.to_list())

                    A_lwe = cmod(A_lwe, self.mlwe.q)
                    B_lwe = cmod(B_lwe, self.mlwe.q)

                else:
                    A = self.mlwe.generate_A(random_bytes)
                    A_lwe = transform_matrix_lwe(A.to_list())
                    error_lwe = transform_vector_lwe(self.mlwe.generate_error(random_bytes).to_list())
                    B_lwe = np.tensordot(A_lwe, self.secret, axes=1) + error_lwe
            
            else:
                A = self.mlwe.generate_A(random_bytes)
                A_lwe = transform_matrix_lwe(A.to_list())
                B_lwe = np.tensordot(A_lwe, self.secret, axes=1)
                if mod_q:
                    A_lwe = cmod(A_lwe, self.mlwe.q)
                    B_lwe = cmod(B_lwe, self.mlwe.q)

            # Save the generated samples
            self.A[i*n*k:(i+1)*n*k, :] = A_lwe
            self.B[i*n*k:(i+1)*n*k] = B_lwe
        
        self.reduced = False

    def initialize_A(self):
        """
        Initializes the A matrix only, without generating B or the secret.
        """
        n = self.params['n']
        k = self.params['k']
        num_gen = self.params['num_gen']

        self.A = np.zeros((num_gen*k*n, k*n))

        for i in range(num_gen):
            random_bytes = self.mlwe.get_random_bytes()

            A = self.mlwe.generate_A(random_bytes)
            A_lwe = transform_matrix_lwe(A.to_list())

            # Save the generated samples
            self.A[i*n*k:(i+1)*n*k, :] = A_lwe

            if self.params['mod_q']:
                self.A = cmod(self.A, self.mlwe.q)

        self.reduced = False

    def initialize_secret(self):
        """
        Initializes the secret key for the LWE scheme.
        """
        if self.A is None:
            raise ValueError("A matrix is not initialized. Please run initialize() or initialize_A() first.")
        
        n = self.params['n']
        k = self.params['k']
        num_gen = self.params['num_gen']
        add_noise = self.params['add_noise']
        mod_q = self.params['mod_q']
        
        self.mlwe = MLWE(self.params)

        random_bytes = self.mlwe.get_random_bytes()
        secret_mlwe = self.mlwe.generate_secret(random_bytes)
        self.secret = transform_vector_lwe(secret_mlwe.to_list())
        self.secret = cmod(self.secret, self.mlwe.q)

        A_split = np.array_split(self.A, num_gen)
        self.B = np.zeros((num_gen *n*k))

        for i in range(num_gen):
            random_bytes = self.mlwe.get_random_bytes()
            
            if add_noise:
                error_lwe = transform_vector_lwe(self.mlwe.generate_error(random_bytes).to_list())
                B_lwe = np.tensordot(A_split[i], self.secret, axes=1) + error_lwe
                if mod_q:
                    B_lwe = cmod(B_lwe, self.mlwe.q)
            else:
                B_lwe = np.tensordot(A_split[i], self.secret, axes=1)
                if mod_q:
                    B_lwe = cmod(B_lwe, self.mlwe.q)

            # Save the generated samples
            self.B[i*n*k:(i+1)*n*k] = B_lwe

        if self.reduced: 
            A_to_reduce = np.stack([self.A[ind] for ind in self.indices])
            B_to_reduce = np.stack([self.B[ind] for ind in self.indices])

            self.RA = cmod(self.R @ A_to_reduce, self.mlwe.q)

            self.RB = cmod(self.R @ B_to_reduce[:, :, np.newaxis], self.mlwe.q)
            self.RB = np.squeeze(self.RB, axis=-1)

            self.non_zero_indices = np.any(self.RA != 0, axis=2)

    def get_indices_to_reduce(self):
        """
        Prepares the matrices to be reduced based on the parameters.
        """
        n = self.params['n']
        k = self.params['k']

        if self.params['reduction_samples'] is None:
            block_sizes = self.params['bkz_block_sizes']
            if isinstance(block_sizes, str):
                block_sizes = parse_range(block_sizes)
            last_block_size = block_sizes[-1]
            delta_0 = get_hermite_root_factor(last_block_size)
            m = get_optimal_sample_size(n * k, self.mlwe.q, self.params['penalty'], delta_0)
        elif 0 <= self.params['reduction_samples'] <= 1:
            m = int(self.params['reduction_samples'] * n * k)
            if m == 0:
                # Special case where need to process a row at a time
                m = 1
        else:
            m = self.params['reduction_samples']

        n_rows_matrix = m + n * k

        num_gen = self.params['num_gen']

        if not self.params['reduction_resampling']:
            # Split the dataset into chunks of size m
            split_num = self.A.shape[0] // m
            remainder = self.A.shape[0] % m

            indices_vectors = [np.arange(i*m, (i+1)*m) for i in range(split_num)]
            if remainder > 0:
                last_indices = np.arange(split_num*m, split_num*m + remainder)
                # If not enough rows, sample additional rows from A to reach size m
                if remainder < m:
                    needed = m - remainder
                    extra_indices = np.random.choice(self.A.shape[0], size=needed, replace=False)
                    last_indices = np.concatenate([last_indices, extra_indices])

                indices_vectors.append(last_indices)
        else:
            indices_vectors = []
            total_rows = self.A.shape[0]
            if self.params['seed'] is not None:
                np.random.seed(self.params['seed'])
            
            if self.params['num_matrices'] > 0:
                min_trials = self.params['num_matrices']
            else:
                min_samples = self.params['min_samples']
                if min_samples is None:
                    min_samples = 0

                min_trials = calculate_min_trials(n*k*num_gen, m, target_prob=0.99, max_k=n*k*num_gen)
                min_trials = max(min_trials, min_samples // n_rows_matrix + 1)

            for _ in range(min_trials):
                indices = np.random.choice(total_rows, size=m, replace=False)
                indices_vectors.append(indices)
        
        indices_vectors = np.stack(indices_vectors)

        return indices_vectors

    def reduction(self):
        """
        Reduces the dataset using the reduction matrix.
        """
        self.indices = self.get_indices_to_reduce()
        A_to_reduce = np.stack([self.A[ind] for ind in self.indices])
        
        num_blocks = len(self.indices)
        
        # Use ThreadPoolExecutor to parallelize the reduction process
        print(f"Reducing {num_blocks} matrices using {get_slurm_cpu_count()} threads.")
        args = []
        for i, mat in enumerate(A_to_reduce):
            params_copy = self.params.copy()
            params_copy['checkpoint_filename'] = params_copy['checkpoint_filename'] + f"_{i}.npy"
            reduction = Reduction(params_copy)
            args.append((reduction, mat))

        # Use multiprocessing to parallelize the reduction process
        with ProcessPoolExecutor(max_workers=get_slurm_cpu_count()) as executor:
            self.R = np.stack(list(executor.map(LWEDataset.reduce_wrapper, args)))

        # Compute reduced A and B with a single call
        self.RA = cmod(self.R @ A_to_reduce, self.mlwe.q)

        if self.B is not None:
            B_to_reduce = np.stack([self.B[ind] for ind in self.indices])
            self.RB = cmod(self.R @ B_to_reduce[:, :, np.newaxis], self.mlwe.q)
            self.RB = np.squeeze(self.RB, axis=-1)

        self.non_zero_indices = np.any(self.RA != 0, axis=2) 

        self.reduced = True

    @staticmethod
    def reduce_wrapper(args):
        """
        Wrapper function for the reduction process.
        """
        reduction, matrix = args
        return reduction.reduce(matrix)

    def attack(self, 
               attack_strategy="tour",
               attack_every=1,

               save_strategy="no",
               save_every=None,

               stop_strategy="no",
               stop_after=None,
               
               save_at_the_end=True
               ):
        """ Reduces the dataset using the reduction matrix. """
        if self.A is None:
            raise ValueError("A matrix must be initialized before running the attack. Please run initialize() or initialize_A() first.")
        
        if self.B is None and attack_every is not None:
            raise ValueError("B vector must be initialized before running the attack with attack_every. Please run initialize_secret() first.")

        if attack_strategy not in ["tour", "time", "no", "hour", "minute", "second"]:
            raise ValueError("Invalid attack strategy. Choose from 'tour', 'time', or 'no'.")
        if save_strategy not in ["tour", "time", "no", "hour", "minute", "second"]:
            raise ValueError("Invalid save strategy. Choose from 'tour', 'time', or 'no'.")
        
        if attack_strategy in ["tour", "time", "hour", "minute", "second"] and attack_every is None:
            raise ValueError("attack_every must be specified when using attack_strategy 'tour' or 'time'.")
        
        if save_strategy in ["tour", "time", "hour", "minute", "second"] and save_every is None:
            raise ValueError("save_every must be specified when using save_strategy 'tour' or 'time'.")

        if self.indices is None:
            self.indices = self.get_indices_to_reduce()

        A_to_reduce = np.stack([self.A[ind] for ind in self.indices])
        
        if attack_strategy != "no":
            B_to_reduce = np.stack([self.B[ind] for ind in self.indices])

        num_matrices = A_to_reduce.shape[0]

        is_rlwe = self.params['k'] == 1 and self.params['reduction_samples'] == 1 and not self.params['reduction_resampling']

        # Use ThreadPoolExecutor to parallelize the reduction process
        if self.params['verbose']:
            print(f"Attacking {num_matrices} matrices using {get_slurm_cpu_count()} threads.")

        args = []
        if self.R is not None:
            RA = cmod(self.R @ A_to_reduce, self.mlwe.q)

        for i in range(num_matrices):
            reduction = ContinuousReduction(self.params)
            if self.RC is not None:
                # If RC is already computed, use it to initialize the reduction
                reduction.initialize_matrix(A_to_reduce[i])
                _, _, std_b = get_b_distribution(self.params, RA[i], self.R[i])

                reduction.initialize_saved_reduced(
                    vectors=self.best_RC[i].copy() if self.best_RC is not None else self.RC[i].copy(),
                    priorities=std_b
                )

                args.append([reduction.to_state_dict(), self.RC[i].copy(), 1])
            else:
                # First attack
                args.append([reduction.to_state_dict(), A_to_reduce[i].copy(), 1])

        if is_rlwe:
            A_to_reduce = A_to_reduce[:, np.newaxis, :, :]
            if attack_strategy != "no":
                B_to_reduce = B_to_reduce[:, np.newaxis, :]

        self.reduced = True
        tour = 0
        start_time = time.time()
        previous_reduction_time = self.reduction_time

        if attack_strategy == "hour":
            attack_every *= 3600
            attack_strategy = "time"
        elif attack_strategy == "minute":
            attack_every *= 60
            attack_strategy = "time"
        elif attack_strategy == "second":
            attack_strategy = "time"

        if save_strategy == "hour":
            save_every *= 3600
            save_strategy = "time"
        elif save_strategy == "minute":
            save_every *= 60
            save_strategy = "time"
        elif save_strategy == "second":
            save_strategy = "time"

        if stop_strategy == "hour":
            stop_after *= 3600
            stop_strategy = "time"
        elif stop_strategy == "minute":
            stop_after *= 60
            stop_strategy = "time"
        elif stop_strategy == "second":
            stop_strategy = "time"

        last_save_time = start_time if save_strategy == "time" else None
        last_attack_time = start_time if attack_strategy == "time" else None

        n_jobs = get_slurm_cpu_count()
        n_jobs = min(n_jobs, num_matrices)  # Limit the number of jobs to the number of matrices
        current_time = start_time
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            while True:
                tour += 1

                # Run parallel reduction
                R_reduced = []
                RC_reduced = []
                for i, (R, arg) in enumerate(executor.map(LWEDataset.continuous_reduction_wrapper, args)):
                    args[i] = arg
                    R_reduced.append(R)
                    RC_reduced.append(arg[1])

                self.R = pad_vectors_to_max(R_reduced).astype(int)
                self.RC = np.stack(RC_reduced).astype(int)

                if is_rlwe:
                    # Reduction was made using the RLWE/MLWE structure, so we can use the reduction circulants
                    self.R = np.stack([np.stack([neg_circ(row).T for row in reduced_matrix]) for reduced_matrix in self.R]).astype(int)
                    
                current_time = time.time()

                # Check if it's time to save
                if save_strategy == "time" and current_time - last_save_time >= save_every:
                    # Save the reduced matrices and best matrices for further reduction
                    if self.params['reduction_max_size'] > 0:
                        self.best_RC = pad_vectors_to_max([np.stack([item[2] for item in arg[0]['priority_queue']['heap']]) for arg in args]).astype(int)
                    else:
                        self.best_RC = np.stack([arg[0]['best_matrix'] for arg in args]).astype(int)

                    self.reduction_time = previous_reduction_time + current_time - start_time

                    self.save_reduced(postfix=f'_{(current_time - start_time) // save_every}')
                    last_save_time = current_time
                elif save_strategy == "tour" and tour % save_every == 0:
                    # Save the reduced matrices and best matrices for further reduction
                    if self.params['reduction_max_size'] > 0:
                        self.best_RC = pad_vectors_to_max([np.stack([item[2] for item in arg[0]['priority_queue']['heap']]) for arg in args]).astype(int)
                    else:
                        self.best_RC = np.stack([arg[0]['best_matrix'] for arg in args]).astype(int)

                    self.reduction_time = previous_reduction_time + current_time - start_time

                    self.save_reduced(postfix=f'_{tour // save_every}')

                self.RA = cmod(self.R @ A_to_reduce, self.mlwe.q).astype(int)

                self.non_zero_indices = np.any(self.RA != 0, axis=2)

                if self.params['verbose']:
                    reduction_factor = np.mean(np.std(self.RA[self.non_zero_indices], axis=-1)) / np.mean(np.std(A_to_reduce, axis=-1))
                    std_b = np.mean(self.get_b_distribution()[2])
                    prob = std_to_prob(std_b, self.mlwe.q)
                    print(f"Tour {tour} | Time: {current_time - start_time:.2f}s | Mean std_B: {std_b:.2f} | Reduction Factor: {reduction_factor:.4f} | Prob: {prob:.4f}")

                # Check if it's time to attack
                if attack_strategy == "time" and current_time - last_attack_time >= attack_every or \
                    attack_strategy == "tour" and tour % attack_every == 0:
                    
                    if attack_strategy == "time":
                        last_attack_time = current_time

                    self.RB = cmod(self.R @ B_to_reduce[..., np.newaxis], self.mlwe.q).astype(int)
                    self.RB = np.squeeze(self.RB, axis=-1)

                    found, guessed_secret = self.train()
                    if found and self.params['verbose']:
                        attack_time = time.time() - start_time
                        print(f"Secret found after {attack_time:.2f} seconds.")
                        report(self.secret, guessed_secret)

                        if save_at_the_end:
                            # Save the reduced matrices and best matrices for further reduction
                            self.RC = np.stack([arg[1] for arg in args])

                            if self.params['reduction_max_size'] > 0:
                                self.best_RC = pad_vectors_to_max([np.stack([item[2] for item in arg[0]['priority_queue']['heap']]) for arg in args]).astype(int)
                            else:
                                self.best_RC = np.stack([arg[0]['best_matrix'] for arg in args]).astype(int)

                            self.reduction_time = previous_reduction_time + current_time - start_time
                            self.save_reduced()

                        return guessed_secret, attack_time
                
                if stop_strategy == "time" and current_time - start_time >= stop_after:
                    if self.params['verbose']:
                        print(f"Stopping after {stop_after} seconds.")
                    break
                elif stop_strategy == "tour" and tour >= stop_after:
                    if self.params['verbose']:
                        print(f"Stopping after {stop_after} tours.")
                    break
            
        if self.params['verbose'] and attack_strategy != "no":
            print(f"Attack completed after {time.time() - start_time:.2f} seconds.")
        
        if save_at_the_end:
            # Save the reduced matrices and best matrices for further reduction
            self.RC = np.stack([arg[1] for arg in args])
            
            if self.params['reduction_max_size'] > 0:
                self.best_RC = pad_vectors_to_max([np.stack([item[2] for item in arg[0]['priority_queue']['heap']]) for arg in args]).astype(int)
            else:
                self.best_RC = np.stack([arg[0]['best_matrix'] for arg in args]).astype(int)

            self.reduction_time = previous_reduction_time + current_time - start_time

            self.save_reduced()
        
        return None, time.time() - start_time

    @staticmethod
    def continuous_reduction_wrapper(args):
        """ Wrapper function for the reduction process. """
        reduction_state, matrix_to_reduce, times = args
        reduction = ContinuousReduction.from_state_dict(reduction_state)
        start_time = time.time()
        R, matrix_to_reduce = reduction.reduce(matrix_to_reduce, times=times)
        tour_time = time.time() - start_time
        
        # Update the times the reduction tour will be repeated
        if tour_time < 60 * 2: # Less than 2 minute -> increase by 1
            times += 1
        elif tour_time >= 60 * 10: # More than 10 minutes -> decrease by 1
            times -= 1
            if times < 1:
                times = 1
        elif tour_time >= 60 * 20: # More than 20 minutes -> reset to 1
            times = 1
        
        return R, [reduction.to_state_dict(), matrix_to_reduce, times]

    
    def train(self):
        """ 
        Trains the model on the dataset using the reduction matrix.
        If percentages are not provided, it uses the default percentages.
        """
        if not isinstance(self.params['train_percentages'], list):
            self.params['train_percentages'] = [self.params['train_percentages']]

        # After each reduction, it tries to retrive the secret
        std_B = self.approximate_b()
        sorted_indices = np.argsort(std_B)

        A_reduced = self.get_A()
        best_b = self.best_b

        if self.params['verbose']:
            b_real = get_no_mod(self.params, A_reduced, self.secret, self.get_B())
        
        for p in self.params['train_percentages']:
            # Select top N% with the lowest std
            num_selected = int(len(std_B) * p)
            selected_indices = sorted_indices[:num_selected]

            if self.params['verbose']:
                exact_candidates = np.sum(best_b[selected_indices] == b_real[selected_indices])
                total_selection = len(selected_indices)
                print(f"[BEST {int(p*100)}% STD] True B is the best candidate: {exact_candidates} / {total_selection} ({100 * exact_candidates / total_selection:.2f}%)")

            found, guessed_secret = train_model(self,
                                                A = A_reduced[selected_indices],
                                                b = best_b[selected_indices])
            if found:
                return True, guessed_secret
            elif np.all(guessed_secret == self.secret):
                print("Warning: Guessed secret is equal to the real secret, but it is not checked correctly.")
                return True, guessed_secret
            
        return False, None

    def get_A(self):
        """ Returns the A matrix. """
        if self.A is None:
            raise ValueError("A matrix is not initialized. Please run initialize() or initialize_A() first.")
        
        return self.RA[self.non_zero_indices] if self.reduced else self.A
    
    def get_B(self):
        """ Returns the B vector. """
        if self.B is None:
            raise ValueError("B vector is not initialized. Please run initialize_secret() first.")
        
        return self.RB[self.non_zero_indices] if self.reduced else self.B

    def get_secret(self):
        """ Returns the secret vector. """
        if self.secret is None:
            raise ValueError("Secret vector is not initialized. Please run initialize_secret() first.")
        
        return self.secret

    def get_error_distribution(self):
        if self.reduced:
            return get_error_distribution(self.params, self.R[self.non_zero_indices])
        else:
            return get_error_distribution(self.params)

    def get_b_distribution(self):
        if self.reduced:
            return get_b_distribution(self.params, self.RA[self.non_zero_indices], self.R[self.non_zero_indices])
        else:
            return get_b_distribution(self.params, self.A)

    def approximate_b(self):
        """
        Returns the approximate distribution of B values based on A and secret and error distributions.
        """
        expected_B, _, std_B = self.get_b_distribution()

        self.b_candidates, self.b_probs = compute_b_candidates_and_probs(
            b_mod = self.get_B() % self.mlwe.q,
            mu = expected_B,
            sigma = std_B, 
            modulus = self.mlwe.q, 
            num_std = self.params['approximation_std'], 
            threshold = self.params['approximation_threshold']
        )

        self.best_b = np.array([self.b_candidates[i][np.argmax(probs)] for i, probs in enumerate(self.b_probs)])
        return std_B
    
    def save_reduced(self, postfix=''):
        """ Save the dataset to a file using pickle. """

        if not self.reduced:
            raise ValueError("Dataset is not reduced. Please run reduction() or attack() first.")
        
        data_to_save = {
            'params': self.params,
            'A': self.A,
            'secret': self.secret,
            'B': self.B,
            'RC': self.RC,
            'best_RC': self.best_RC,
            'indices': self.indices,
            'reduction_time': self.reduction_time
        }

        filename = get_filename_from_params(self.params, ext=postfix + '.pkl')

        # Save the dataset to a file
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)

        if self.params['verbose']:
            print(f"Dataset saved to {filename}")

    @classmethod
    def load_reduced(cls, filepath):
        """ Loads the dataset from a file. """

        # Load the dataset from a file
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        # Create a new instance of the class
        dataset = cls(loaded_data['params'])

        dataset.secret = loaded_data['secret']
        dataset.B = loaded_data['B']
        dataset.A = loaded_data['A']


        if 'indices' not in loaded_data:
            dataset.R = loaded_data['R']
            dataset.RA = loaded_data['RA']
            dataset.RB = loaded_data['RB']
        else:
            dataset.indices = loaded_data['indices']
            A_to_reduce = np.stack([dataset.A[ind] for ind in dataset.indices])

            if 'RC' in loaded_data:
                dataset.RC = loaded_data['RC']
                try:
                    dataset.best_RC = loaded_data['best_RC']
                
                    if dataset.params['matrix_config'] in ['salsa', 'dual']:
                        m = A_to_reduce.shape[1]
                        dataset.R = np.stack([reduced_matrix[:, :m] / loaded_data['params']['penalty'] for reduced_matrix in dataset.best_RC])
                    else:
                        n = A_to_reduce.shape[2]
                        dataset.R = np.stack([reduced_matrix[:, n:] / loaded_data['params']['penalty'] for reduced_matrix in dataset.best_RC])

                    if dataset.params['k'] == 1 and dataset.params['reduction_samples'] == 1 and not dataset.params['reduction_resampling']:
                        dataset.R = np.stack([np.stack([neg_circ(row).T for row in reduced_matrix]) for reduced_matrix in dataset.R])
                        A_to_reduce = A_to_reduce[:, np.newaxis, :, :]
                except:
                    print("Warning: 'best_RC' corrupted. Using 'RC' instead.")
                    dataset.best_RC = None
                    if dataset.params['matrix_config'] in ['salsa', 'dual']:
                        m = A_to_reduce.shape[1]
                        dataset.R = np.stack([reduced_matrix[:, :m] / loaded_data['params']['penalty'] for reduced_matrix in dataset.RC])
                    else:
                        n = A_to_reduce.shape[2]
                        dataset.R = np.stack([reduced_matrix[:, n:] / loaded_data['params']['penalty'] for reduced_matrix in dataset.RC])
            else:
                dataset.R = loaded_data['R']

            if 'reduction_time' in loaded_data:
                dataset.reduction_time = loaded_data['reduction_time']
            
            dataset.RA = cmod(dataset.R @ A_to_reduce, dataset.mlwe.q)

            dataset.non_zero_indices = np.any(dataset.RA != 0, axis=-1)

            if dataset.B is not None:
                B_to_reduce = np.stack([dataset.B[ind] for ind in dataset.indices])
                if dataset.params['k'] == 1 and dataset.params['reduction_samples'] == 1 and not dataset.params['reduction_resampling']:
                    B_to_reduce = B_to_reduce[:, np.newaxis, :]

                dataset.RB = cmod(dataset.R @ B_to_reduce[..., np.newaxis], dataset.mlwe.q)
                dataset.RB = np.squeeze(dataset.RB, axis=-1)

        dataset.reduced = True

        return dataset
    
    @classmethod
    def load_params_from_file(cls, filepath):
        """
        Loads the parameters from a file.
        """
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        return loaded_data['params']

    @classmethod
    def load_reduced_from_salsa(cls, data_path, top_percent=1.0):
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

        # Create a new instance of the class
        dataset = cls(params)

        if params['k'] != 1:
            orig_A_path = data_path / f"origA_n{params['n']}_k{params['k']}_logq{int(np.round(np.log2(params['q'])))}.npy"
        else:
            orig_A_path = data_path / f"origA_n{params['n']}_logq{int(np.round(np.log2(params['q'])))}.npy"

        dataset.A = np.load(orig_A_path)
        dataset.params['num_gen'] = dataset.A.shape[0] // (params['n'] * params['k'])
        dataset.reduced = True

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
                        RA = cmod(R @ dataset.A[indices], dataset.mlwe.q)
                        non_zero_indices = np.any(RA != 0, axis=-1)
                        _, _, std_B = get_b_distribution(dataset.params, RA[non_zero_indices], R[non_zero_indices])
                        sorted_indices = np.argsort(std_B)[:num_rows]
                        R = R[sorted_indices]

                    full_indices.append(indices)
                    full_R.append(R)
                    indices, RT = [], []

        dataset.R = np.stack(full_R)
        dataset.indices = np.stack(full_indices)

        A_to_reduce = np.stack([dataset.A[ind] for ind in dataset.indices])
        dataset.RA = cmod(dataset.R @ A_to_reduce, dataset.mlwe.q)
        dataset.non_zero_indices = np.any(dataset.RA != 0, axis=-1)

        return dataset
