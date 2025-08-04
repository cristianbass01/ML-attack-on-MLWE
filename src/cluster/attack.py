# attack.py
import sys, os, argparse, json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from ml_attack.dataset import LWEDataset
from ml_attack.utils import get_lwe_default_params, get_continuous_reduction_default_params, get_filename_from_params, get_train_default_params, parse_range, cbd_expected_hamming_weight

from collections import Counter

def get_hw_range(params, args):
    n = params['n']
    if args.hw_range:
        return parse_range(args.hw_range)
    if params['secret_type'] == 'binary':
        return range(1, n // 2 + 1)
    elif params['secret_type'] == 'ternary':
        return range(1, 2 * (n // 3) + 1)
    elif params['secret_type'] == 'cbd':
        return range(1, cbd_expected_hamming_weight(n, params['eta']) + 1)
    else:
        return []

def main(updated_params, args):
    params = get_lwe_default_params()
    params.update(get_continuous_reduction_default_params())
    params.update(get_train_default_params())
    params.update(updated_params)
    
    args_dict = vars(args)
    args_dict.pop('params', None)  # Remove 'params' from args dictionary
    print(f"Parameters: {params}")
    print(f"Arguments: {args_dict}")

    attack_times = []
    for i in range(args.num_attacks):
        if args.reload or args.reload_from is not None:
            if args.reload_from:
                filename = args.reload_from
            else:
                filename = get_filename_from_params(params)

            if os.path.exists(filename):
                print(f"Reloading dataset from {filename} with updated parameters.")
                dataset = LWEDataset.load_reduced(filename)
                dataset.params.update(updated_params)
                print(f"Parameters after reload: {dataset.params}")
            else:
                print(f"Dataset not found at {filename}. Exiting.")
                sys.exit(1)
        else:
            dataset = LWEDataset(params)
            dataset.initialize()
        
        # Attack mode
        _, attack_time = dataset.attack(
            attack_strategy=args.attack_strategy,
            attack_every=args.attack_every,
            save_strategy=args.save_strategy,
            save_every=args.save_every,
            stop_strategy=args.stop_strategy,
            stop_after=args.stop_after,
            save_at_the_end=args.save_at_the_end,
        )

        args.save_strategy = "no"  # Save only the first attack

        print(f"Attack {i+1}/{args.num_attacks} completed in {attack_time:.2f} seconds.")
        attack_times.append(attack_time)

        if 'seed' in params and params['seed'] is not None and args.num_attacks > 1:
            params['seed'] += 1  # Increment seed for each attack

    if len(attack_times) > 1:
        mean_attack_time = np.mean(attack_times)
        std_attack_time = np.std(attack_times)

        print(f"Mean attack time: {mean_attack_time}")
        print(f"Standard deviation of attack time: {std_attack_time}")

    if args.num_attacks == 0:
        if args.reload_from is not None:
            filename = args.reload_from
        else:
            filename = get_filename_from_params(params)
        
        if args.reload_from_salsa is not None:
            dataset = LWEDataset.load_reduced_from_salsa(filename, top_percent=args.reload_from_salsa)
        else:
            dataset = LWEDataset.load_reduced(filename)
        
        dataset.params.update(updated_params)
        print(f"Reloaded dataset from {filename} with parameters: {dataset.params}")

    if len(args.train_secret_types) > 0:
        print("Performing attack on all secret types.")
        preprocessed_time = dataset.reduction_time
        print(f"Preprocessing time: {preprocessed_time:.2f} seconds")

        n = dataset.params['n']

        for secret_type in args.train_secret_types:
            dataset.params['secret_type'] = secret_type
            dataset.params['seed'] = None
            dataset.params['eta'] = 2
            if secret_type == 'cbd':
                dataset.params['error_type'] = 'cbd'
            else:
                dataset.params['error_type'] = 'gaussian'

            choosen_hw = get_hw_range(dataset.params, args)

            success_counter = Counter()
            train_counter = Counter()
            for hw in choosen_hw:
                dataset.params['hw'] = hw
                for _ in range(args.num_training_repeats):
                    dataset.initialize_secret()
                    real_hw = dataset.get_hamming_weight()
                    train_counter[real_hw] += 1

                    found, _ = dataset.train()
                    if found:
                        success_counter[real_hw] += 1

                success_rates = [
                    f"{hw}: {success_counter[hw]}/{train_counter[hw]} ({success_counter[hw] / train_counter[hw] if train_counter[hw] > 0 else 0:.2f})"
                    for hw in sorted(train_counter.keys())
                ]
                print(", ".join(success_rates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True, help='JSON string of parameters')
    parser.add_argument('--num_attacks', type=int, default=1, help='Number of attacks to perform')
    parser.add_argument('--attack_strategy', type=str, default='tour', help='Attack strategy to use')
    parser.add_argument('--attack_every', type=int, default=1, help='Attack every N samples')
    parser.add_argument('--save_strategy', type=str, default='no', help='Save strategy to use')
    parser.add_argument('--save_every', type=int, default=1, help='Save every N samples')
    parser.add_argument('--stop_strategy', type=str, default='no', help='Stop strategy to use')
    parser.add_argument('--stop_after', type=int, default=48, help='Stop after N samples')
    parser.add_argument('--save_at_the_end', action='store_true', help='Save the dataset at the end of the attack')
    parser.add_argument('--reload', action='store_true', help='Reload the dataset from disk if it exists')
    parser.add_argument('--reload_from', type=str, default=None, help='Reload dataset from a specific file if it exists')
    parser.add_argument('--train_secret_types', nargs='+', default=[], help='Secret types to train on')
    parser.add_argument('--reload_from_salsa', type=float, default=None, help='Reload dataset from Salsa with top 1% of samples')
    parser.add_argument('--num_training_repeats', type=int, default=10, help='Number of training repeats for each hw of each secret type')
    parser.add_argument('--hw_range', type=str, default=None, help='Specific range of hw to try, formatted as start:end:step (e.g., "1:10:1")')

    args = parser.parse_args()

    try:
        updated_params = json.loads(args.params)
        main(updated_params, args)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)


