import os
import numpy as np
from fpylll import FPLLL, LLL, BKZ, GSO, IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from subprocess import Popen, PIPE

from ml_attack.utils import get_b_distribution

FLOAT_UPGRADE = {
    "double": "long double",
    "long double": "dd",
    "dd": "qd",
    "qd": "mpfr_250",
}
MAX_TIME_BKZ = 60  # 1 minute

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


def calc_std(X, Q, m):
    # mat is in right half of matrix
    mat = X[:, m:] % Q
    mat[mat > Q // 2] -= Q
    return np.sqrt(12) * np.std(mat[np.any(mat != 0, axis=1)]) / Q


# flatter functions
def encode_intmat(intmat):
    """will put in expected format for flatter input."""
    fplll_Ap_encode = "[" + intmat.__str__() + "]"
    fplll_Ap_encode = fplll_Ap_encode.encode()
    return fplll_Ap_encode


def decode_intmat(out):
    """Decodes output intmat from flatter and puts it back in np.array form."""
    t_str = out.rstrip().decode()[1:-1]
    Ap = np.array(
        [np.array(line[1:-1].split(" ")).astype(int) for line in t_str.split("\n")[:-1]]
    )
    return Ap


class Reduction(object):
    def __init__(self, params: dict):
        """
        Initialize the reduction class.
        :param params: A dictionary of parameters for the reduction:
            - q (int): The modulus for the reduction.
            - lookback (int): Number of steps over which to calculate the average decrease for stalling detection. Default: 2.
            - float_type (str): The floating-point precision type to use. Default: "double".
            - algos (list): List of algorithms to use in the reduction process. Default: ["flatter", "BKZ2.0"].
            - bkz_block_sizes (list): List of block sizes for the BKZ algorithm. Default: [30, 40].
            - bkz_deltas (list): List of delta values for the BKZ algorithm. Default: [0.96, 0.99].
            - flatter_alphas (list): List of alpha values for the Flatter algorithm. Default: [0.04, 0.025].
            - penalty (int): Penalty factor for arranging the reduction matrix. Default: 10.
            - verbose (bool): Boolean flag to enable or disable verbose logging. Default: False.
            - checkpoint_filename (str): Path to save the best reduction checkpoint. Default: "./best_reduction.npy".
            - reload_checkpoint (bool): Boolean flag to reload the checkpoint. Default: False.
        """

        # Some basic checks
        self.params = params
        self.q = params["q"]
        assert self.q > 0, "Must provide a valid modulus q."

        # Set up parameters.
        self.set_float_type(params["float_type"])

        # Set up function calls.
        self.algos = params["algos"]
        self.algo_idx = 0

        self.std_tracker = []
        self.lookback = params['lookback']  # number of steps over which to calculate (avg) decrease, must run given algo at least this many times before switching.

        if isinstance(self.lookback, int):
            self.lookback = [self.lookback] * len(self.algos)
        elif len(self.lookback) != len(self.algos):
            raise ValueError("lookback must be either an int or a list of ints with the same length as algos.")
        
        self.min_decrease = -0.001 # min decrease we have to see over self.lookback steps to consider it "working".
        self.num_std = params['reduction_std']

        self.bkz_block_sizes = params["bkz_block_sizes"]
        self.bkz_block_size_idx = 0

        self.bkz_deltas = params["bkz_deltas"]
        self.bkz_delta_idx = 0

        self.flatter_alphas = params["flatter_alphas"]
        self.flatter_alpha_idx = 0

        self.penalty = params["penalty"]
        self.verbose = params["verbose"]

        self.checkpoint_filename = params["checkpoint_filename"]
        self.reload_checkpoint = params["reload_checkpoint"]


    def run_algo(self, algo_idx, Ap):
        """
        Run the specified algorithm on the input matrix.
        :param algo_idx: The index of the algorithm to run.
        :param Ap: The input matrix to be reduced.penalty
        :return: The reduced matrix.
        """
        algo = self.algos[algo_idx]
        match algo:
            case "flatter":
                Ap = self.run_flatter_once(Ap)
            case "BKZ":
                Ap = self.run_bkz_once(Ap)
            case "BKZ2.0":
                Ap = self.run_bkz2_once(Ap)
            case "LLL":
                Ap = self.run_lll_once(Ap)
            case _:
                raise ValueError(f"Unknown algorithm: {algo}, available: ['flatter', 'BKZ', 'BKZ2.0', 'LLL']")
            
        return Ap
        
    def upgrade_delta(self):
        """
        Increase the delta value for the next round of BKZ.
        """
        if self.bkz_delta_idx < len(self.bkz_deltas) - 1:
            self.bkz_delta_idx += 1
            self.bkz_delta = self.bkz_deltas[self.bkz_delta_idx]
            return True
        else:
            return False
        
    def upgrade_block_size(self):
        """
        Increase the block size for the next round of BKZ.
        """
        if self.bkz_block_size_idx < len(self.bkz_block_sizes) - 1:
            self.bkz_block_size_idx += 1
            self.bkz_block_size = self.bkz_block_sizes[self.bkz_block_size_idx]
            return True
        else:
            return False
        
    def upgrade_alpha(self):
        """
        Increase the alpha value for the next round of Flatter.
        """
        if self.flatter_alpha_idx < len(self.flatter_alphas) - 1:
            self.flatter_alpha_idx += 1
            self.flatter_alpha = self.flatter_alphas[self.flatter_alpha_idx]
            return True
        else:
            return False
        
    def upgrade_algo(self):
        """
        Switch to the next algorithm.
        """
        self.algo_idx = (self.algo_idx + 1) % len(self.algos)
        self.steps_same_algo = 0

        if self.algo_idx == 0:
            return False
        else:
            return True
    
    def log(self, message):
        """
        Log a message to the console.
        """
        if self.verbose:
            print(message)
    
    def get_R(self, matrix_to_reduce):
        """
        Get the R matrix from the reduction.
        """
        if self.penalty != 0:
            R = matrix_to_reduce[:, :self.m] / self.penalty
        else:
            R = matrix_to_reduce

        return R

    def set_float_type(self, float_type):
        self.float_type = float_type
        parsed_float_type = float_type.split("_")
        if len(parsed_float_type) == 2:
            self.float_type, precision = parsed_float_type
            assert self.float_type == "mpfr"
            FPLLL.set_precision(int(precision))

    def arrange_reduction_matrix(self, matrix_to_reduce):
        # Arrange the matrix as [0 q*Id; w*Id A]
        matrix_to_reduce %= self.q
        matrix_to_reduce[matrix_to_reduce > self.q // 2] -= self.q
        m, n = self.m, self.n

        if self.penalty != 0:
            A_red = np.zeros((m + n, m + n), dtype=int)
            A_red[n:, :m] = np.identity(m, dtype=np.int64) * self.penalty
            A_red[n:, m:] = matrix_to_reduce
            A_red[:n, m:] = np.identity(n, dtype=int) * self.q
        else:
            A_red = np.zeros((m + n, m), dtype=int)
            A_red[n:, :] = matrix_to_reduce
            A_red[:n, :] = np.identity(n, dtype=int) * self.q

        return A_red  # Ap.shape = (m+N)*(m+N)

    def run_flatter_once(self, Ap):
        """
        Runs a single loop of flatter.
        """
        self.log(f"Starting new flatter run.")
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        fplll_Ap_encoded = encode_intmat(fplll_Ap)
        try:
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            p = Popen(["flatter", "-alpha", str(self.alpha)], stdin=PIPE, stdout=PIPE, env=env)
        except Exception as e:
            print(f"flatter failed with error {e}")
        out, _ = p.communicate(input=fplll_Ap_encoded)  # output from the flatter run.
        Ap = decode_intmat(out)

        return Ap

    def run_bkz_once(self, Ap):
        """
        Runs a single round of BKZ.
        """
        self.log(f"Starting new BKZ run.")
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
        bkz_params = BKZ.Param(self.bkz_block_size, delta=self.bkz_delta, max_time=MAX_TIME_BKZ)
        
        L = LLL.Reduction(M, delta=self.bkz_delta)
        BKZ_Obj = BKZ.Reduction(M, L, bkz_params)
        
        # Run once.    
        BKZ_Obj()

        Ap = np.zeros((Ap.shape[0], Ap.shape[1]), dtype=np.int64)
        fplll_Ap.to_matrix(Ap)

        return Ap
    
    def run_bkz2_once(self, Ap):
        """
        Runs a single round of BKZ.
        """
        self.log(f"Starting new BKZ2.0 run.")
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        bkz_params = BKZ.Param(self.bkz_block_size, delta=self.bkz_delta, max_time=MAX_TIME_BKZ)
        
        # Run once.
        while True:
            try:
                M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
                BKZ_Obj = BKZ2(M)
                BKZ_Obj(bkz_params)
                break
            except Exception as e:
                print(e)
                # for bkz2.0, this would catch the case where it needs more precision for floating point arithmetic
                # for bkz, the package does not throw the error properly. Make sure to start with enough precision
                if self.float_type in FLOAT_UPGRADE:
                    self.set_float_type(FLOAT_UPGRADE[self.float_type])
                    print(f"Upgrading float type to {self.float_type}")
                else:
                    print(f"Error running bkz2.0. No more float types to upgrade to.")
                    break

        Ap = np.zeros((Ap.shape[0], Ap.shape[1]), dtype=np.int64)
        fplll_Ap.to_matrix(Ap)

        return Ap

    def run_lll_once(self, Ap):
        # TODO this should run the LLL algo from fpylll
        raise NotImplementedError("This is not implemented yet.")
    
    def solvable(self, A_red, A):
        """
        Check if the lwe is solvable
        """
        R = self.get_R(A_red)

        # Compute RA
        RA = np.tensordot(R, A, axes=1) % self.q
        RA[RA > (self.q // 2)] -= self.q
        
        _, _, std_b = get_b_distribution(self.params, RA, R) 

        # Check if the reduction is solvable
        conditions = (self.num_std * std_b) < (self.q / 2)

        self.log(f" - Solvable for: {np.sum(conditions)} out of {len(conditions)}")
        return np.all(conditions)

    def is_on_stall(self):
        if len(self.std_tracker) > self.lookback[self.algo_idx] and self.steps_same_algo > self.lookback[self.algo_idx]:
            decreases = [
                self.std_tracker[i] - self.std_tracker[i-1]
                for i in range(-self.lookback[self.algo_idx], -1)
            ]
            if np.mean(decreases) > self.min_decrease:
                return True  # Your mean decrease is higher than the mandated minimum decrease over the last self.lookback rounds - you've stalled.
        return False

    def control(self, Ap):
        # Run checks.
        algo = self.algos[self.algo_idx]

        # Compute the current std
        current_std = calc_std(Ap, self.q, self.m)
        self.std_tracker.append(current_std)
        self.log(f" - Current std: {current_std}")

        # Check if it got better and save the checkpoint
        if current_std == min(self.std_tracker):
            np.save(self.checkpoint_filename, Ap)
            self.log(f" - Checkpoint saved to {self.checkpoint_filename}")

        # Check if we stalled
        stop = False
        if self.is_on_stall():

            if self.upgrade_algo():
                new_algo = self.algos[self.algo_idx]
                self.log(f"Stalling... Switching from {algo} to {new_algo}.")
            else:
                # Upgrading params to 
                upgraded = self.upgrade_alpha()
                if not upgraded:
                    upgraded = self.upgrade_block_size()
                    upgraded = self.upgrade_delta() or upgraded

                if upgraded:
                    new_algo = self.algos[self.algo_idx]
                    self.log(f"Stalling... Upgrading parameters and switching back to {new_algo}.")
                else:
                    self.log(f"No more parameters to upgrade. Terminating...")
                    stop = True
                
        return stop

    def reduce(self, matrix_to_reduce):
        # Tracking params
        self.std_tracker = []
        self.steps_same_algo = 0

        # Preprocess matrix to reduce
        self.m, self.n = matrix_to_reduce.shape

        initial_matrix = matrix_to_reduce.copy() % self.q
        initial_matrix[initial_matrix > self.q // 2] -= self.q

        if self.reload_checkpoint:
            if os.path.exists(self.checkpoint_filename):
                self.log(f"Loading checkpoint from {self.checkpoint_filename}")
                matrix_to_reduce = np.load(self.checkpoint_filename)
            else:
                self.log(f"Checkpoint file {self.checkpoint_filename} not found. Starting from scratch.")
                matrix_to_reduce = self.arrange_reduction_matrix(matrix_to_reduce)
        else:
            matrix_to_reduce = self.arrange_reduction_matrix(matrix_to_reduce)

        # Parameter settings
        self.bkz_block_size_idx = 0
        self.bkz_delta_idx = 0
        self.flatter_alpha_idx = 0
        self.algo_idx = 0

        self.bkz_block_size = self.bkz_block_sizes[self.bkz_block_size_idx]
        self.bkz_delta = self.bkz_deltas[self.bkz_delta_idx]
        self.alpha = self.flatter_alphas[self.flatter_alpha_idx]

        # Save the initial std for tracking.
        starting_std = calc_std(matrix_to_reduce, self.q, self.m)
        self.log(f" - Starting std: {starting_std}")
        self.std_tracker.append(starting_std)
            
        # Run the reduction
        while True:
            # Run the current algorithm.
            matrix_to_reduce = self.run_algo(self.algo_idx, matrix_to_reduce)
            self.steps_same_algo += 1

            # Interleave with polish
            matrix_to_reduce = polish(matrix_to_reduce)

            # Check if the reduction is solvable
            if self.solvable(matrix_to_reduce, initial_matrix):
                self.log("Reduction is solvable.")
                break

            # Run checks.
            if self.control(matrix_to_reduce):
                break
            
        # Load the best matrix from the checkpoint is it exists
        if os.path.exists(self.checkpoint_filename):
            self.log(f"Loading best matrix from checkpoint {self.checkpoint_filename}")
            matrix_to_reduce = np.load(self.checkpoint_filename)

        # Remove the checkpoint file if it exists
        if os.path.exists(self.checkpoint_filename):
            os.remove(self.checkpoint_filename)

        # Get R from the best saved matrix
        R = self.get_R(matrix_to_reduce)

        return R