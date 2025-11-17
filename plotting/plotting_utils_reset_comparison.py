import os
import re
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class NBFLLogAnalyzer:
    """Centralized class for analyzing NBFL logs and generating plots with reset comparison"""
    
    def __init__(self, log_base_path='/Users/chenhang/Documents/Working/NBFL/logs'):
        self.log_base_path = log_base_path
        self.attack_type_map = {
            0: 'No Attack', 
            1: 'Poison Attack', 
            2: 'Label Flipping Attack', 
            3: 'Lazy Attack', 
            4: 'Poison & Lazy'
        }
        
    def parse_folder_name(self, folder_name):
        """Parse folder name to extract configuration parameters"""
        # Pattern to match NBFL folder structure with reset
        nbfl_pattern = r'NBFL_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_reset_(\d+)'
        nbfl_match = re.match(nbfl_pattern, folder_name)
        
        if nbfl_match:
            seed, data_dist, alpha, ndevices, nsamples, rounds, mal, attack, reset = nbfl_match.groups()
            config_key = f"NBFL_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack}_reset_{reset}"
            return {
                'config_key': config_key,
                'seed': int(seed),
                'data_dist': data_dist,
                'alpha': alpha,
                'ndevices': int(ndevices),
                'nsamples': int(nsamples),
                'rounds': int(rounds),
                'mal': int(mal),
                'attack': int(attack),
                'reset': int(reset),
                'method': 'NBFL'
            }
        
        # Pattern to match baseline methods with reset
        baseline_patterns = {
            'STANDALONE': r'STANDALONE_LTH_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_reset_(\d+)',
            'FEDAVG': r'FEDAVG_NO_PRUNE_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_reset_(\d+)',
            'LOTTERYFL': r'LotteryFL_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_reset_(\d+)',
            'POIS': r'PoIS_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_reset_(\d+)',
            'CELL': r'CELL_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_reset_(\d+)'
        }
        
        for method, pattern in baseline_patterns.items():
            match = re.match(pattern, folder_name)
            if match:
                seed, data_dist, alpha, ndevices, nsamples, rounds, mal, attack, reset = match.groups()
                config_key = f"{method}_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack}_reset_{reset}"
                return {
                    'config_key': config_key,
                    'seed': int(seed),
                    'data_dist': data_dist,
                    'alpha': alpha,
                    'ndevices': int(ndevices),
                    'nsamples': int(nsamples),
                    'rounds': int(rounds),
                    'mal': int(mal),
                    'attack': int(attack),
                    'reset': int(reset),
                    'method': method
                }
        
        return None

    def group_logs_by_config(self):
        """Group log folders by configuration, ignoring seeds and timestamps"""
        logs_dir = f'{self.log_base_path}'
        config_groups = defaultdict(list)
        
        for folder in os.listdir(logs_dir):
            folder_path = f'{logs_dir}/{folder}'
            if os.path.isdir(folder_path):
                parsed = self.parse_folder_name(folder)
                if parsed:
                    # Group by configuration, keeping method-specific keys
                    if parsed['method'] == 'NBFL':
                        config_key = f"NBFL_mnist_{parsed['data_dist']}_alpha_{parsed['alpha']}_ndevices_{parsed['ndevices']}_nsamples_{parsed['nsamples']}_rounds_{parsed['rounds']}_mal_{parsed['mal']}_attack_{parsed['attack']}_reset_{parsed['reset']}"
                    else:
                        # Use method-specific config key for baselines
                        config_key = f"{parsed['method']}_mnist_{parsed['data_dist']}_alpha_{parsed['alpha']}_ndevices_{parsed['ndevices']}_nsamples_{parsed['nsamples']}_rounds_{parsed['rounds']}_mal_{parsed['mal']}_attack_{parsed['attack']}_reset_{parsed['reset']}"
                    
                    config_groups[config_key].append({
                        'path': folder_path,
                        'seed': parsed['seed'],
                        'config': parsed,
                        'method': parsed['method']
                    })
        
        return config_groups

    def get_unique_configs_without_reset(self, config_groups):
        """Extract unique configurations from grouped logs, ignoring reset value"""
        unique_configs = set()
        for config_key, logs in config_groups.items():
            if logs and 'NBFL_' in config_key:  # Only consider NBFL configs
                config = logs[0]['config']  # Get config from first log in group
                # Don't include reset in the unique config tuple
                unique_configs.add((config['mal'], config['attack'], config['alpha'], config['data_dist']))
        return unique_configs
    
    def get_available_alpha_values(self, config_groups=None):
        """Extract all available alpha values from the log folders"""
        if config_groups is None:
            config_groups = self.group_logs_by_config()
        
        alpha_values = set()
        for config_key, logs in config_groups.items():
            if logs:
                config = logs[0]['config']
                alpha_values.add(config['alpha'])
        
        return sorted(list(alpha_values))
    
    def should_skip_config(self, mal, attack_type):
        """Check if configuration should be skipped based on attack/mal combinations"""
        return (attack_type == 0 and mal != 0) or (attack_type != 0 and mal == 0)

    def load_logger_data(self, log_path, logger_concerning):
        """Load logger data from pickle file"""
        try:
            with open(f'{log_path}/logger.pickle', 'rb') as file:
                logger = pickle.load(file)
                if logger_concerning not in logger:
                    print(f"Warning: {logger_concerning} not found in {log_path}")
                    return None
                return logger
        except Exception as e:
            print(f"Error loading {log_path}: {e}")
            return None

    def process_metric_data(self, logger, logger_concerning, legitimate_only=False, mal_count=0):
        """Process metric data from logger (works for accuracy and sparsity metrics)"""
        metric_values_over_devices = []
        
        try:
            for comm_round, metric_dict in logger[logger_concerning].items():
                if legitimate_only and 'local' in logger_concerning:
                    # Sort devices and take only legitimate ones (exclude last mal_count devices)
                    sorted_devices = sorted(metric_dict.keys())
                    legitimate_devices = sorted_devices[:-mal_count] if mal_count > 0 else sorted_devices
                    metric_values = [metric_dict[device] for device in legitimate_devices]
                else:
                    metric_values = list(metric_dict.values())
                metric_values_over_devices.append(metric_values)
            
            if metric_values_over_devices:
                metric_values_over_devices = list(zip(*metric_values_over_devices))
                result = np.mean(metric_values_over_devices, axis=0)
                
                # Ensure we always return an array, even if it's 1D
                if np.isscalar(result) or result.ndim == 0:
                    return np.array([result])
                return result
            else:
                print("\033[91m" + f"    Debug: No metric data found for {logger_concerning}" + "\033[0m")
                return None
                
        except Exception as e:
            print("\033[91m" + f"    Debug: Error in process_metric_data: {e}" + "\033[0m")
            return None

    def process_logs_for_config(self, logs_for_config, logger_concerning, legitimate_only=False, mal_count=0):
        """Process all logs for a specific configuration"""
        avg_values_over_runs = []
        
        for log_info in logs_for_config:
            log_path = log_info['path']
            seed = log_info['seed']
            
            print(f"  Processing seed {seed} from {log_path}")
            logger = self.load_logger_data(log_path, logger_concerning)
            if logger is None:
                print(f"    Failed to load logger data")
                continue
                
            avg_over_devices = self.process_metric_data(logger, logger_concerning, legitimate_only, mal_count)
            if avg_over_devices is not None:
                # Check for NaN values
                if np.any(np.isnan(avg_over_devices)):
                    print(f"    Warning: NaN values detected in seed {seed}, skipping")
                    continue
                
                avg_values_over_runs.append(avg_over_devices)
                print(f"    Successfully processed seed {seed}: {len(avg_over_devices)} rounds")
            else:
                print(f"    No valid data for seed {seed}")
        
        return avg_values_over_runs

    def generate_metric_plots_with_reset_comparison(self, logger_concerning, y_axis_label, 
                                                    alpha_filter=None, legitimate_plots=False, 
                                                    verbose=True, include_baselines=False):
        """Generate plots comparing reset_0 vs reset_1 for each configuration"""
        if verbose:
            print(f"\nGenerating RESET COMPARISON plots for {logger_concerning}...")
        
        config_groups = self.group_logs_by_config()
        
        # Auto-detect alpha values if not specified
        if alpha_filter is None:
            alpha_filter = self.get_available_alpha_values(config_groups)
            if verbose:
                print(f"Auto-detected alpha values: {alpha_filter}")
        elif verbose:
            print(f"Using specified alpha values: {alpha_filter}")
        
        # Get unique configs without reset (so we can compare reset_0 vs reset_1)
        unique_configs = self.get_unique_configs_without_reset(config_groups)
        
        for mal, attack_type, alpha, data_dist in unique_configs:
            # Skip invalid combinations
            if self.should_skip_config(mal, attack_type):
                continue
            
            # Apply alpha filter
            if alpha not in alpha_filter:
                continue
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing: mal={mal}, attack={attack_type}, alpha={alpha}, data_dist={data_dist}")
                print(f"Comparing reset_0 vs reset_1")
                print(f"{'='*60}")
            
            # Create a single plot comparing reset_0 and reset_1
            plt.figure(figsize=(6, 4))
            
            # Process both reset values
            reset_data = {}
            for reset in [0, 1]:
                if verbose:
                    print(f"\n--- Processing reset={reset} ---")
                
                # Find matching NBFL configuration key
                nbfl_config_key = f"NBFL_mnist_{data_dist}_alpha_{alpha}_ndevices_20_nsamples_200_rounds_25_mal_{mal}_attack_{attack_type}_reset_{reset}"
                
                if nbfl_config_key not in config_groups:
                    print(f"No NBFL logs found for config: {nbfl_config_key}")
                    continue
                
                nbfl_logs = config_groups[nbfl_config_key]
                
                if not nbfl_logs:
                    print(f"No logs available for reset={reset}")
                    continue
                
                if verbose:
                    print(f"Found {len(nbfl_logs)} logs for NBFL reset={reset}")
                
                # Process logs for NBFL
                avg_values_over_runs = self.process_logs_for_config(
                    nbfl_logs, logger_concerning, legitimate_only=False, mal_count=mal
                )
                
                if not avg_values_over_runs:
                    if verbose:
                        print(f"No valid NBFL data found for reset={reset}")
                    continue
                
                if verbose:
                    print(f"NBFL reset={reset}: Successfully processed {len(avg_values_over_runs)} runs")
                
                # Calculate mean and std across different seeds
                mean_line = np.mean(avg_values_over_runs, axis=0)
                std = np.std(avg_values_over_runs, axis=0)
                
                # Check for NaN or empty data
                if np.any(np.isnan(mean_line)) or len(mean_line) == 0:
                    if verbose:
                        print(f"Skipping reset={reset}: invalid mean_line (NaN or empty)")
                    continue
                
                reset_data[reset] = {
                    'mean_line': mean_line,
                    'std': std,
                    'num_runs': len(avg_values_over_runs)
                }
            
            # Plot both reset values if we have data for both
            if not reset_data:
                if verbose:
                    print("No data available for any reset value, skipping plot")
                plt.close()
                continue
            
            # Define colors and line styles for reset comparison
            reset_colors = {0: 'red', 1: 'blue'}
            reset_labels = {0: 'Without Parameters Reset', 1: 'With Parameters Reset'}
            
            for reset, data in reset_data.items():
                mean_line = data['mean_line']
                std = data['std']
                num_runs = data['num_runs']
                color = reset_colors[reset]
                label = reset_labels[reset]
                
                # Plot with error bands
                plt.fill_between(range(1, len(mean_line) + 1), mean_line - std, mean_line + std,
                                facecolor=color, color=color, alpha=0.2)
                
                plot_label = f'{label} ({num_runs} seeds)'
                plt.plot(range(1, len(mean_line) + 1), mean_line, color=color, 
                        linestyle='-', linewidth=2, label=plot_label)
                
                # Add text annotations every 5 x-axis ticks
                x = 0
                plt.text(x + 1, mean_line[x], f'{mean_line[x]:.2f}', 
                        ha='center', va='bottom', fontsize=8, color=color)
                for x in range(4, len(mean_line), 5):
                    plt.text(x + 1, mean_line[x], f'{mean_line[x]:.2f}', 
                            ha='center', va='bottom', fontsize=8, color=color)
            
            plt.legend(loc='best', fontsize=10)
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel(y_axis_label, fontsize=12)
            
            # Extend y-axis range to ensure flat lines at 1.0 are visible
            if y_axis_label == 'Sparsity':
                plt.ylim(-0.05, 1.05)
            
            # Create title and filename
            title = f'{" ".join(logger_concerning.split("_")).title()} - {mal} Atkers - {self.attack_type_map[attack_type]}, α: {alpha}, {data_dist.upper()}\nWithout Parameters Reset vs With Parameters Reset'
            filename = f'{self.log_base_path}/reset_comparison_{logger_concerning}_mal_{mal}_attack_{attack_type}_alpha_{alpha}_{data_dist}.png'
            
            plt.title(title, fontsize=12)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot: {filename}")
            plt.clf()
            plt.close()
            
            # Generate legitimate device plots if requested
            if legitimate_plots and 'local' in logger_concerning and attack_type != 0 and mal > 0:
                plt.figure(figsize=(6, 4))
                
                legitimate_reset_data = {}
                for reset in [0, 1]:
                    nbfl_config_key = f"NBFL_mnist_{data_dist}_alpha_{alpha}_ndevices_20_nsamples_200_rounds_25_mal_{mal}_attack_{attack_type}_reset_{reset}"
                    
                    if nbfl_config_key not in config_groups:
                        continue
                    
                    nbfl_logs = config_groups[nbfl_config_key]
                    if not nbfl_logs:
                        continue
                    
                    legitimate_avg_values = self.process_logs_for_config(
                        nbfl_logs, logger_concerning, legitimate_only=True, mal_count=mal
                    )
                    
                    if legitimate_avg_values:
                        legit_mean_line = np.mean(legitimate_avg_values, axis=0)
                        legit_std = np.std(legitimate_avg_values, axis=0)
                        
                        legitimate_reset_data[reset] = {
                            'mean_line': legit_mean_line,
                            'std': legit_std,
                            'num_runs': len(legitimate_avg_values)
                        }
                
                if legitimate_reset_data:
                    for reset, data in legitimate_reset_data.items():
                        mean_line = data['mean_line']
                        std = data['std']
                        num_runs = data['num_runs']
                        color = reset_colors[reset]
                        label = reset_labels[reset]
                        
                        plt.fill_between(range(1, len(mean_line) + 1), mean_line - std, mean_line + std,
                                        facecolor=color, color=color, alpha=0.2)
                        
                        plot_label = f'{label} ({num_runs} seeds)'
                        plt.plot(range(1, len(mean_line) + 1), mean_line, color=color, 
                                linestyle='-', linewidth=2, label=plot_label)
                        
                        # Add text annotations
                        x = 0
                        plt.text(x + 1, mean_line[x], f'{mean_line[x]:.2f}', 
                                ha='center', va='bottom', fontsize=8, color=color)
                        for x in range(4, len(mean_line), 5):
                            plt.text(x + 1, mean_line[x], f'{mean_line[x]:.2f}', 
                                    ha='center', va='bottom', fontsize=8, color=color)
                    
                    plt.legend(loc='best', fontsize=10)
                    plt.xlabel('Communication Round', fontsize=12)
                    plt.ylabel(y_axis_label, fontsize=12)
                    
                    legit_title = f'LEGIT {" ".join(logger_concerning.split("_")).title()} - {mal} Atkers - {self.attack_type_map[attack_type]}, α: {alpha}, {data_dist.upper()}\nWithout Parameters Reset vs With Parameters Reset'
                    legit_filename = f'{self.log_base_path}/reset_comparison_{logger_concerning}_mal_{mal}_attack_{attack_type}_alpha_{alpha}_{data_dist}_legitimate.png'
                    
                    plt.title(legit_title, fontsize=12)
                    plt.savefig(legit_filename, dpi=300, bbox_inches='tight')
                    print(f"✓ Saved legitimate plot: {legit_filename}")
                    plt.clf()
                    plt.close()
                else:
                    plt.close()
