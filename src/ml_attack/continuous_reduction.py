import os
import numpy as np
from fpylll import FPLLL, LLL, BKZ, GSO, IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from subprocess import Popen, PIPE

from ml_attack.utils import get_b_distribution, parse_range, polish, get_optimal_vector_norm, cmod
from ml_attack.priority_queue import BoundedPriorityQueue
import time

FLOAT_UPGRADE = {
    "d": "ld",
    "ld": "dd",
    "dd": "qd",
    "qd": "mpfr_150",
    "mpfr_150": "mpfr_200",
    "mpfr_200": "mpfr_250",
    "mpfr_250": "mpfr_300",
    "mpfr_300": "mpfr_350"
}
MAX_TIME_BKZ = 60  # 1 minute

class ContinuousReduction(object):
    def __init__(self, params: dict):
        """
        Initialize the reduction class.
        :param params: A dictionary of parameters for the reduction:
            - q (int): The modulus for the reduction.
            - lookback (int): Number of steps over which to calculate the average decrease for stalling detection. Default: 2.
            - float_type (str): The floating-point precision type to use. Default: "double".
            - algos (list): List of algorithms to use in the reduction process. Default: ["flatter", "BKZ2.0"].
            - bkz_block_sizes (range): List of block sizes for the BKZ algorithm. Default: [30, 40].
            - bkz_deltas (range): List of delta values for the BKZ algorithm. Default: [0.96, 0.99].
            - flatter_alpha (list): List of alpha values for the Flatter algorithm. Default: [0.04, 0.025].
            - penalty (int): Penalty factor for arranging the reduction matrix. Default: 10.
            - verbose (bool): Boolean flag to enable or disable verbose logging. Default: False.
            - checkpoint_filename (str): Path to save the best reduction checkpoint. Default: "./best_reduction.npy".
            - reload_checkpoint (bool): Boolean flag to reload the checkpoint. Default: False.
        """

        # Some basic checks
        self.params = params.copy()
        self.q = self.params["q"]

        # Set up parameters.
        self.set_float_type(self.params["float_type"])

        self.no_improvements = 0
        self.n_stall = 0
        self.lookback = self.params["lookback"]

        # Warm-up using flatter
        self.flatter_countdown = self.params["warmup_steps"]
        self.flatter_alpha = self.params["flatter_alpha"]

        # Then update to BKZ2.0
        self.bkz_delta = self.params["bkz_delta"]
        self.bkz_block_sizes = self.params["bkz_block_sizes"]
        if isinstance(self.bkz_block_sizes, str):
            self.bkz_block_sizes = parse_range(self.bkz_block_sizes)
        self.bkz_block_size_idx = 0

        self.use_polish = self.params["use_polish"]
        self.interleaved_steps = self.params["interleaved_steps"]

        # if pnjBKZ is used
        # self.crossover = params["crossover"]

        self.initial_matrix = None
        if self.params['reduction_max_size'] > 0:
            self.saved_reduced = BoundedPriorityQueue(self.params["reduction_max_size"])
            self.use_priority = True
        else:
            self.saved_reduced = None
            self.saved_stds = None
            self.use_priority = False

        self.penalty = self.params["penalty"]
        self.verbose = self.params["verbose"]

        self.steps_same_algo = 0
        self._first_pnj_bkz = True
        self._first_bkz = True
        self.m = None  # Number of rows in the input matrix
        self.n = None  # Number of columns in the input matrix

        self.matrix_config = self.params["matrix_config"]

    def run_algo(self, Ap):
        """
        Run the specified algorithm on the input matrix.
        :param Ap: The input matrix to be reduced.penalty
        :return: The reduced matrix.
        """
        if self.flatter_countdown > 0:
            self.flatter_countdown -= 1
            if self.flatter_countdown == 0:
                self.n_stall = 0
                self.no_improvements = 0
                self.steps_same_algo = 0
                
            return self.run_flatter_once(Ap)
        #elif self.crossover < 0 or self.bkz_block_sizes[self.bkz_block_size_idx] < self.crossover: 
        #    return self.run_bkz2_once(Ap)
        #else:
        #    return self.run_pnj_bkz_once(Ap)
        else:
            return self.run_bkz2_once(Ap)
        
    def log(self, message):
        """ Log a message to the console. """
        if self.verbose:
            print(message)
    
    def get_R(self, matrix_to_reduce):
        """ Get the R matrix from the reduction. """    
        if self.matrix_config in ["salsa", "dual"]:
            # Matrix in the form [wR, RA + qC]
            return cmod(matrix_to_reduce[:, :self.m] / self.penalty, self.q)
        elif self.matrix_config == "original":
            # Matrix in the form [RA + qC, wR]
            return cmod(matrix_to_reduce[:, self.n:] / self.penalty, self.q)
        else:
            raise ValueError(f"Unknown matrix configuration: {self.matrix_config}. Supported: 'salsa', 'dual', 'original'.")

    def set_float_type(self, float_type):
        self.float_type = float_type
        parsed_float_type = float_type.split("_")
        if len(parsed_float_type) == 2:
            self.float_type, precision = parsed_float_type
            assert self.float_type == "mpfr"
            FPLLL.set_precision(int(precision))

    def arrange_reduction_matrix(self, matrix_to_reduce):
        m, n = self.m, self.n
        # Check if the matrix is 1-dimensional
        A_red = np.zeros((m + n, m + n), dtype=int)
        if self.matrix_config == "salsa":
            # Matrix in form [0 q*In; w*Im A]
            A_red[n:, :m] = np.identity(m, dtype=int) * self.penalty
            A_red[n:, m:] = matrix_to_reduce
            A_red[:n, m:] = np.identity(n, dtype=int) * self.q
        elif self.matrix_config == "dual":
            # Matrix in form [w*Im A; 0 q*In]
            A_red[:m, :m] = np.identity(m, dtype=int) * self.penalty
            A_red[:m, m:] = matrix_to_reduce
            A_red[m:, m:] = np.identity(n, dtype=int) * self.q
        elif self.matrix_config == "original":
            # Matrix in form [A  w*Im; 0 q*In]
            A_red[:m, n:] = np.identity(m, dtype=int) * self.penalty
            A_red[:m, :n] = matrix_to_reduce
            A_red[m:, :n] = np.identity(n, dtype=int) * self.q
        else:
            raise ValueError(f"Unknown matrix configuration: {self.matrix_config}. Supported: 'salsa', 'dual', 'original'.")

        return A_red  # Ap.shape = (m+N)*(m+N)

    def run_flatter_once(self, Ap):
        """ Runs a single loop of flatter. """
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        
        fplll_Ap_encoded = "[" + fplll_Ap.__str__() + "]"
        fplll_Ap_encoded = fplll_Ap_encoded.encode()

        try:
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            p = Popen(["flatter", "-alpha", str(self.flatter_alpha)], stdin=PIPE, stdout=PIPE, env=env)
        except Exception as e:
            print(f"flatter failed with error {e}")
        out, _ = p.communicate(input=fplll_Ap_encoded)  # output from the flatter run.

        t_str = out.rstrip().decode()[1:-1]
        Ap = np.array([np.array(line[1:-1].split(" ")).astype(int) for line in t_str.split("\n")[:-1]]).astype(int)
        return Ap
    
    def run_bkz2_once(self, Ap):
        """ Runs a single round of BKZ. """
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        if self._first_bkz:
            LLL.reduction(fplll_Ap)
            self._first_bkz = False
        
        bkz_block_size = self.bkz_block_sizes[self.bkz_block_size_idx]
        bkz_params = BKZ.Param(bkz_block_size, delta=self.bkz_delta, max_time=MAX_TIME_BKZ)
        
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

        Ap = np.zeros((Ap.shape[0], Ap.shape[1]), dtype=int)
        fplll_Ap.to_matrix(Ap)
        return Ap

# The following code is commented out because it is not used in the current implementation.

# from g6k.algorithms.bkz import pump_n_jump_bkz_tour
# from g6k.siever import Siever
# from g6k.utils.stats import SieveTreeTracer

    #def run_pnj_bkz_once(self, Ap):
    #    """Runs a single round of pnjBKZ."""
    #    self.log(f"- Running pnjBKZ with block size {self.bkz_block_sizes[self.bkz_block_size_idx]}...")
    #    fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
    #    g6k = Siever(fplll_Ap)
    #    if self._first_pnj_bkz:
    #        g6k.lll(0, g6k.full_n)
    #        self._first_pnj_bkz = False
    #    else:
    #        g6k.initialize_local(0, g6k.full_n, False)
    #    self.log(f"- LLL with full_n={g6k.full_n} completed.")
    #    # Initialize the tracer
    #    tracer = SieveTreeTracer(g6k, root_label=("pnjBKZ"), start_clocks=True)
    #    # Extract block size and set jump and dim4free_fun
    #    block_size = self.bkz_block_sizes[self.bkz_block_size_idx]
    #    jump = 1  # or parameterize this if needed
    #    extra_dim4free = 12
    #    dim4free_fun = lambda x: 11.5 + 0.075 * x  # matches example in G6K
    #    # Run pnjBKZ
    #    pump_n_jump_bkz_tour(
    #        g6k, tracer, block_size,
    #        jump=jump,
    #        extra_dim4free=extra_dim4free,
    #        dim4free_fun=dim4free_fun,
    #        pump_params={"down_sieve": True},  # or parametrize as needed
    #        verbose=self.verbose
    #    )
    #    # Retrieve and return the updated matrix
    #    Ap = np.zeros((fplll_Ap.nrows, fplll_Ap.ncols), dtype=np.int64)
    #    fplll_Ap.to_matrix(Ap)
    #    return Ap
    
    #def run_LLL_once(self, Ap):
    #    """ Runs a single round of LLL. """
    #    self.log(f"- Running LLL...")
    #    fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
    #    try:
    #        LLL.reduction(fplll_Ap)
    #    except Exception as e:
    #        print(f"LLL failed with error {e}")
    #    Ap = np.zeros((Ap.shape[0], Ap.shape[1]), dtype=np.int64)
    #    fplll_Ap.to_matrix(Ap)
    #    return Ap
    
    def control(self, A_red):
        """ Save the best lines of the reduced matrix to a checkpoint file. """
        R = self.get_R(A_red)

        # Compute RA
        RA = cmod(np.tensordot(R, self.initial_matrix, axes=1), self.q)

        _, _, std_b = get_b_distribution(self.params, RA, R)
        
        non_zero_indiced = np.where(std_b > 0)[0]

        algo_name = "flatter" if self.flatter_countdown > 0 else f"bkz2.0_{self.bkz_block_sizes[self.bkz_block_size_idx]}"
        
        if self.use_priority:
            num_updates = self.saved_reduced.add_batch(A_red[non_zero_indiced], std_b[non_zero_indiced])
        elif self.saved_stds is None:
            self.saved_stds = std_b
            self.saved_reduced = A_red
            num_updates = len(std_b)
        else:
            # Identify indices where std_b < self.saved_stds, ignoring std_b items that are 0.
            better_indices = np.where((std_b < self.saved_stds) & (std_b != 0))[0]

            # Identify indices where self.saved_stds is 0 but std_b is not
            zero_best_indices = np.where((self.saved_stds == 0) & (std_b != 0))[0]

            # Combine the indices
            better_indices = np.unique(np.concatenate((better_indices, zero_best_indices)))
            num_updates = len(better_indices)
            if num_updates > 0:
                self.saved_reduced[better_indices] = A_red[better_indices]
                self.saved_stds[better_indices] = std_b[better_indices]

        self.log(f"- Algo: {algo_name} | Updated {num_updates}/{len(non_zero_indiced)} | Mean std_B: {np.mean(std_b[non_zero_indiced]):.2f}")
        
        if num_updates/len(non_zero_indiced) >= 0.1:
            self.n_stall = 0
            self.no_improvements = 0
        else:
            self.n_stall += 1
            if num_updates/len(non_zero_indiced) <= 0.02:
                self.no_improvements += 1
            else:
                self.no_improvements = 0

        # If we stalled for too long, update the algorithm.
        if self.steps_same_algo > self.lookback and self.n_stall >= self.lookback:
            updated = False
            if self.flatter_countdown > 0:
                self.log(f"- Flatter stall after: {self.flatter_countdown}, updating...")
                self.flatter_countdown = 0
                updated = True
            else:
                new_idx = min(self.bkz_block_size_idx + 1, len(self.bkz_block_sizes) - 1)
                if new_idx != self.bkz_block_size_idx:
                    self.log(f"- Updating BKZ block size from {self.bkz_block_sizes[self.bkz_block_size_idx]} to {self.bkz_block_sizes[new_idx]}.")
                    updated = True
                elif self.no_improvements >= self.lookback:
                    self.bkz_block_sizes[self.bkz_block_size_idx] += 2
                    self.log(f"- No improvements for {self.no_improvements} steps, increasing block size to {self.bkz_block_sizes[self.bkz_block_size_idx]}.")
                    updated = True

                self.bkz_block_size_idx = new_idx

                if self.interleaved_steps > 0:
                    self.flatter_countdown = self.interleaved_steps
                    updated = True

            if updated:
                self.steps_same_algo = 0
                self.n_stall = 0
                self.no_improvements = 0

    def initialize_matrix(self, initial_matrix):
        """
        Initialize the reduction with the initial matrix.
        :param initial_matrix: The matrix to be reduced.
        """
        if self.initial_matrix is not None:
            raise ValueError("Initial matrix has already been set.")
        
        # Set the initial matrix and its dimensions
        self.initial_matrix = initial_matrix.copy().astype(int)
        self.m, self.n = initial_matrix.shape
        self.bkz_block_sizes = [blocksize for blocksize in self.bkz_block_sizes if blocksize < self.m + self.n]

    def initialize_saved_reduced(self, vectors, priorities):
        """
        Initialize the saved reduced priority queue with vectors and priorities.
        :param vectors: The vectors to be added to the queue.
        :param priorities: The corresponding priorities for the vectors.
        """
        assert len(vectors) == len(priorities), "Vectors and priorities must match"
        
        if self.use_priority:
            self.saved_reduced.initialize(vectors, priorities)
        else:
            self.saved_reduced = vectors.astype(int)
            self.saved_stds = priorities.astype(int)

    def reduce(self, matrix_to_reduce, times=1):
        """ 
        Reduce the given matrix using the specified algorithms.
        :param matrix_to_reduce: The matrix to be reduced.
        :param times: The number of times to run the reduction algorithms.
        """
        # Preprocess matrix to reduce
        if self.initial_matrix is None:
            self.initialize_matrix(matrix_to_reduce)
            matrix_to_reduce = self.arrange_reduction_matrix(matrix_to_reduce)
            
        # Run the reduction
        start_time = time.time()
        for _ in range(times):
            # Check if the time limit has been exceeded
            if time.time() - start_time > 30 * 60:  # 30 minutes
                self.log("Time limit exceeded. Breaking out of the reduction loop.")
                break

            # Run the current algorithm.
            matrix_to_reduce = self.run_algo(matrix_to_reduce)
            self.steps_same_algo += 1

            # Interleave with polish
            if self.use_polish:
                matrix_to_reduce = polish(matrix_to_reduce)

            # Run checks
            self.control(matrix_to_reduce)

        # Get R from the current matrix to reduce
        if self.use_priority:
            return self.get_R(self.saved_reduced.get_saved_vectors()), matrix_to_reduce
        else:
            return self.get_R(self.saved_reduced), matrix_to_reduce

    def to_state_dict(self):
        """
        Serialize the minimal state of the reduction object.
        Includes necessary parameters and state to resume reduction.
        """
        return {
            "params": self.params,
            "float_type": self.float_type,
            "n_stall": self.n_stall,
            "no_improvements": self.no_improvements,
            "warmup_countdown": self.flatter_countdown,
            "bkz_block_size_idx": self.bkz_block_size_idx,
            "initial_matrix": self.initial_matrix.tolist() if self.initial_matrix is not None else None,
            "priority_queue": self.saved_reduced.to_state_dict() if self.use_priority else None,
            "best_matrix": self.saved_reduced.tolist() if not self.use_priority and self.saved_reduced is not None else None,
            "best_stds": self.saved_stds.tolist() if not self.use_priority and self.saved_stds is not None else None,
            "steps_same_algo": self.steps_same_algo,
            "m": self.m,
            "n": self.n,
        }
    
    @classmethod
    def from_state_dict(cls, state):
        """
        Deserialize a ContinuousReduction object from a state dictionary.
        """
        obj = cls(state["params"])
        obj.set_float_type(state["float_type"])
        obj.n_stall = state["n_stall"]
        obj.no_improvements = state["no_improvements"]
        obj.flatter_countdown = state["warmup_countdown"]
        obj.bkz_block_size_idx = state["bkz_block_size_idx"]
        obj.steps_same_algo = state["steps_same_algo"]
        obj.m = state["m"]
        obj.n = state["n"]

        if state["initial_matrix"] is not None:
            obj.initial_matrix = np.array(state["initial_matrix"])

        if 'priority_queue' in state and state["priority_queue"] is not None:
            obj.saved_reduced = BoundedPriorityQueue.from_state_dict(state["priority_queue"])
        elif state.get("best_matrix") is not None and state.get("best_stds") is not None:
            obj.saved_reduced = np.array(state["best_matrix"])
            obj.saved_stds = np.array(state["best_stds"])
        else:
            obj.saved_reduced = None
            obj.saved_stds = None

        return obj

