import os
import re
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class NBFLLogAnalyzer:
    """Centralized class for analyzing NBFL logs and generating plots"""
    
    def __init__(self, log_base_path='/Users/chenhang/Documents/Working'):
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
        # Pattern to match NBFL folder structure with rewind
        nbfl_pattern = r'NBFL_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_rewind_(\d+)'
        nbfl_match = re.match(nbfl_pattern, folder_name)
        
        if nbfl_match:
            seed, data_dist, alpha, ndevices, nsamples, rounds, mal, attack, rewind = nbfl_match.groups()
            config_key = f"NBFL_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack}_rewind_{rewind}"
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
                'rewind': int(rewind),
                'method': 'NBFL'
            }
        
        # Pattern to match baseline methods with rewind (with actual folder name patterns)
        baseline_patterns = {
            'STANDALONE': r'STANDALONE_LTH_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_rewind_(\d+)',
            'FEDAVG': r'FEDAVG_NO_PRUNE_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_rewind_(\d+)',
            'LOTTERYFL': r'LotteryFL_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_rewind_(\d+)',
            'POIS': r'PoIS_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_rewind_(\d+)',
            'CELL': r'CELL_mnist_seed_(\d+)_(iid|non-iid)_alpha_([\d\.∞]+)_\d{8}_\d{6}_ndevices_(\d+)_nsamples_(\d+)_rounds_(\d+)_mal_(\d+)_attack_(\d+)_rewind_(\d+)'
        }
        
        for method, pattern in baseline_patterns.items():
            match = re.match(pattern, folder_name)
            if match:
                seed, data_dist, alpha, ndevices, nsamples, rounds, mal, attack, rewind = match.groups()
                config_key = f"{method}_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack}_rewind_{rewind}"
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
                    'rewind': int(rewind),
                    'method': method
                }
        
        return None

    def group_logs_by_config(self):
        """Group log folders by configuration, ignoring seeds and timestamps"""
        logs_dir = f'{self.log_base_path}/NBFL/logs'
        config_groups = defaultdict(list)
        
        for folder in os.listdir(logs_dir):
            folder_path = f'{logs_dir}/{folder}'
            if os.path.isdir(folder_path):
                parsed = self.parse_folder_name(folder)
                if parsed:
                    # Create a normalized config key that groups by all parameters except seed
                    # This will allow proper grouping of NBFL, CELL, LotteryFL with same params
                    normalized_config_key = f"{parsed['method']}_mnist_{parsed['data_dist']}_alpha_{parsed['alpha']}_ndevices_{parsed['ndevices']}_nsamples_{parsed['nsamples']}_rounds_{parsed['rounds']}_mal_{parsed['mal']}_attack_{parsed['attack']}_rewind_{parsed['rewind']}"
                    
                    config_groups[normalized_config_key].append({
                        'path': folder_path,
                        'seed': parsed['seed'],
                        'config': parsed,
                        'method': parsed['method']
                    })
        
        return config_groups

    def get_unique_configs(self, config_groups):
        """Extract unique configurations from grouped logs"""
        unique_configs = set()
        for config_key, logs in config_groups.items():
            if logs:
                config = logs[0]['config']  # Get config from first log in group
                # Include all methods in unique configs, not just NBFL
                unique_configs.add((config['mal'], config['attack'], config['alpha'], config['data_dist'], config['rewind'], config['ndevices'], config['nsamples'], config['rounds']))
        return unique_configs

    def get_baseline_logs_for_config(self, config_groups, mal, attack_type, alpha, data_dist, rewind, ndevices, nsamples, rounds):
        """Get baseline method logs for a specific configuration"""
        baseline_logs = {
            'STANDALONE': [],
            'FEDAVG': [],
            'LOTTERYFL': [],
            'POIS': [],
            'CELL': []
        }
        
        # Look for each baseline method with its specific config key
        for method in baseline_logs.keys():
            baseline_config_key = f"{method}_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack_type}_rewind_{rewind}"
            
            if baseline_config_key in config_groups:
                baseline_logs[method] = config_groups[baseline_config_key]
        
        return baseline_logs
    
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
    
    def get_available_data_distributions(self, config_groups=None):
        """Extract all available data distribution types from the log folders"""
        if config_groups is None:
            config_groups = self.group_logs_by_config()
        
        data_dists = set()
        for config_key, logs in config_groups.items():
            if logs:
                config = logs[0]['config']
                data_dists.add(config['data_dist'])
        
        return sorted(list(data_dists))
    
    def get_available_configs_summary(self):
        """Get a summary of all available configurations"""
        config_groups = self.group_logs_by_config()
        unique_configs = self.get_unique_configs(config_groups)
        
        alpha_values = self.get_available_alpha_values(config_groups)
        data_dists = self.get_available_data_distributions(config_groups)
        
        mal_values = sorted(set(config[0] for config in unique_configs))
        attack_values = sorted(set(config[1] for config in unique_configs))
        rewind_values = sorted(set(config[4] for config in unique_configs))
        ndevices_values = sorted(set(config[5] for config in unique_configs))
        nsamples_values = sorted(set(config[6] for config in unique_configs))
        rounds_values = sorted(set(config[7] for config in unique_configs))
        
        return {
            'alpha_values': alpha_values,
            'data_distributions': data_dists,
            'mal_values': mal_values,
            'attack_types': attack_values,
            'rewind_values': rewind_values,
            'ndevices_values': ndevices_values,
            'nsamples_values': nsamples_values,
            'rounds_values': rounds_values,
            'total_configs': len(unique_configs),
            'total_log_groups': len(config_groups)
        }

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
            print("\033[91m" + f"    Debug: logger_concerning: {logger_concerning}" + "\033[0m")
            print("\033[91m" + f"    Debug: logger keys available: {list(logger.keys())}" + "\033[0m")
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

    def create_standard_plot(self, mean_line, std, title, y_axis_label, filename, 
                           color='red', label='NBFL', num_seeds=0, add_annotations=True):
        """Create a standard plot with mean line and std bands"""
        plt.figure(figsize=(10, 6))
        
        # Plot with error bands
        plt.fill_between(range(1, len(mean_line) + 1), mean_line - std, mean_line + std,
                        facecolor=color, color=color, alpha=0.2)
        
        plot_label = f'{label} ({num_seeds} seeds)' if num_seeds > 0 else label
        plt.plot(range(1, len(mean_line) + 1), mean_line, color=color, label=plot_label)
        
        # Add text annotations every 5 x-axis ticks
        if add_annotations:
            x = 0
            plt.text(x + 1, mean_line[x], f'{mean_line[x]:.2f}', 
                    ha='center', va='bottom', fontsize=8, color='black')
            for x in range(4, len(mean_line), 5):
                plt.text(x + 1, mean_line[x], f'{mean_line[x]:.2f}', 
                        ha='center', va='bottom', fontsize=8, color='black')
        
        plt.legend(loc='best')
        plt.xlabel('Communication Round')
        plt.ylabel(y_axis_label)
        plt.title(title)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")
        
        plt.clf()
        plt.close()

    def generate_metric_plots(self, logger_concerning, y_axis_label, 
                            alpha_filter=None, legitimate_plots=False, verbose=True, include_baselines=True):
        """Generate plots for a specific metric across all configurations"""
        if verbose:
            print(f"Generating plots for {logger_concerning}...")
        
        config_groups = self.group_logs_by_config()
        
        # Auto-detect alpha values if not specified
        if alpha_filter is None:
            alpha_filter = self.get_available_alpha_values(config_groups)
            if verbose:
                print(f"Auto-detected alpha values: {alpha_filter}")
        elif verbose:
            print(f"Using specified alpha values: {alpha_filter}")
        
        unique_configs = self.get_unique_configs(config_groups)
        
        for mal, attack_type, alpha, data_dist, rewind, ndevices, nsamples, rounds in unique_configs:
            # Skip invalid combinations
            if self.should_skip_config(mal, attack_type):
                continue
            
            # Apply alpha filter
            if alpha not in alpha_filter:
                continue
            
            if verbose:
                print(f"Processing: mal={mal}, attack={attack_type}, alpha={alpha}, data_dist={data_dist}, rewind={rewind}, ndevices={ndevices}, nsamples={nsamples}, rounds={rounds}")
            
            # Find all available methods for this configuration
            available_methods = []
            all_method_logs = {}
            
            # Check for NBFL logs
            nbfl_config_key = f"NBFL_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack_type}_rewind_{rewind}"
            if nbfl_config_key in config_groups:
                available_methods.append('NBFL')
                all_method_logs['NBFL'] = config_groups[nbfl_config_key]
                if verbose:
                    print(f"Found {len(config_groups[nbfl_config_key])} NBFL log folders with different seeds")
            
            # Check for baseline methods
            if include_baselines:
                baseline_logs = self.get_baseline_logs_for_config(config_groups, mal, attack_type, alpha, data_dist, rewind, ndevices, nsamples, rounds)
                
                if attack_type == 0:
                    # For no attack, include all available baselines
                    methods_to_check = ['STANDALONE', 'FEDAVG', 'LOTTERYFL', 'POIS', 'CELL']
                else:
                    # For attacks, include STANDALONE and other baselines
                    methods_to_check = ['STANDALONE', 'FEDAVG', 'LOTTERYFL', 'POIS', 'CELL']
                
                # Check which baseline methods have actual log files
                for method in methods_to_check:
                    if baseline_logs.get(method):
                        available_methods.append(method)
                        all_method_logs[method] = baseline_logs[method]
                        if verbose:
                            print(f"  Found {len(baseline_logs[method])} {method} logs")
                    else:
                        if verbose:
                            print(f"  No {method} logs found for this config")
            
            # Skip if no methods are available
            if not available_methods:
                if verbose:
                    print(f"  No logs found for any method in this configuration, skipping")
                continue
            
            methods = available_methods
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown'][:len(methods)]
            
            plt.figure(figsize=(10, 6))
            
            # Create method display name mapping
            method_display_names = {
                'NBFL': 'NBFL',
                'STANDALONE': 'Standalone',
                'FEDAVG': 'FedAvg',
                'LOTTERYFL': 'LotteryFL',
                'POIS': 'PoIS',
                'CELL': 'CELL'
            }
            
            # First pass: collect all mean lines to detect overlaps
            method_data = []
            for i, method in enumerate(methods):
                logs_for_method = all_method_logs.get(method, [])
                
                if not logs_for_method:
                    if verbose:
                        print(f"  Skipping {method}: no logs available")
                    continue
                
                if verbose:
                    print(f"  Processing {method} with {len(logs_for_method)} logs")
                
                # Process logs for current method
                avg_values_over_runs = self.process_logs_for_config(
                    logs_for_method, logger_concerning, legitimate_only=False, mal_count=mal
                )
                
                if not avg_values_over_runs:
                    if verbose:
                        print(f"  No valid {method} data found after processing")
                    continue
                
                if verbose:
                    print(f"  {method}: Successfully processed {len(avg_values_over_runs)} runs")
                
                # Calculate mean and std across different seeds
                mean_line = np.mean(avg_values_over_runs, axis=0)
                std = np.std(avg_values_over_runs, axis=0)
                
                # Check for NaN or empty data
                if np.any(np.isnan(mean_line)) or len(mean_line) == 0:
                    if verbose:
                        print(f"  Skipping {method}: invalid mean_line (NaN or empty)")
                    continue
                
                # Check for flat lines at extremes
                if np.all(mean_line == 1.0):
                    if verbose:
                        print(f"  Warning: {method} has constant sparsity=1.0 (flat line at top)")
                elif np.all(mean_line == 0.0):
                    if verbose:
                        print(f"  Warning: {method} has constant sparsity=0.0 (flat line at bottom)")
                
                method_data.append({
                    'method': method,
                    'display_name': method_display_names.get(method, method),
                    'mean_line': mean_line,
                    'std': std,
                    'color': colors[i],
                    'num_runs': len(avg_values_over_runs)
                })
            
            # Smart overlap detection
            def detect_overlaps(method_data, tolerance=1e-6):
                """Detect which methods have overlapping curves"""
                overlap_groups = []
                for i, data1 in enumerate(method_data):
                    group = [i]
                    for j, data2 in enumerate(method_data[i+1:], i+1):
                        # Check if curves are approximately the same
                        if len(data1['mean_line']) == len(data2['mean_line']):
                            if np.allclose(data1['mean_line'], data2['mean_line'], atol=tolerance):
                                group.append(j)
                    if len(group) > 1:
                        overlap_groups.append(group)
                        # Remove already grouped indices from future checks
                        method_data = [d for k, d in enumerate(method_data) if k not in group[1:]]
                return overlap_groups
            
            overlap_groups = detect_overlaps(method_data)
            
            # Assign line styles based on overlaps
            line_styles = ['-'] * len(method_data)  # Default to solid lines
            alternative_styles = ['--', '-.', ':', '-', '--']  # Dashed, dash-dot, dotted, solid, dashed
            
            for group in overlap_groups:
                if verbose:
                    overlapping_methods = [method_data[i]['display_name'] for i in group]
                    print(f"  Overlap detected: {overlapping_methods}")
                
                # Assign different line styles to overlapping methods
                for j, idx in enumerate(group):
                    line_styles[idx] = alternative_styles[j % len(alternative_styles)]
            
            # Second pass: plot with appropriate line styles
            for i, data in enumerate(method_data):
                method = data['method']
                display_name = data['display_name']
                mean_line = data['mean_line']
                std = data['std']
                color = data['color']
                num_runs = data['num_runs']
                line_style = line_styles[i]
                
                # Plot with error bands
                plt.fill_between(range(1, len(mean_line) + 1), mean_line - std, mean_line + std,
                                facecolor=color, color=color, alpha=0.2)
                
                plot_label = f'{display_name} ({num_runs} seeds)'
                plt.plot(range(1, len(mean_line) + 1), mean_line, color=color, 
                        linestyle=line_style, linewidth=2, label=plot_label)
                
                # Add text annotations every 5 x-axis ticks
                x = 0
                plt.text(x + 1, mean_line[x], f'{mean_line[x]:.2f}', 
                        ha='center', va='bottom', fontsize=8, color='black')
                for x in range(4, len(mean_line), 5):
                    plt.text(x + 1, mean_line[x], f'{mean_line[x]:.2f}', 
                            ha='center', va='bottom', fontsize=8, color='black')
            
            plt.legend(loc='best')
            plt.xlabel('Communication Round')
            plt.ylabel(y_axis_label)
            
            # Extend y-axis range to ensure flat lines at 1.0 are visible
            if y_axis_label == 'Sparsity':
                plt.ylim(-0.05, 1.05)  # Add margin above 1.0 for sparsity plots
            
            # Create title and filename
            title = f'{" ".join(logger_concerning.split("_")).title()} - {mal} Atkers - {self.attack_type_map[attack_type]}, α: {alpha}, {data_dist.upper()}, rewind: {rewind}, n_samples: {nsamples}'
            
            # Check if we have any actual data to plot
            if not method_data:
                if verbose:
                    print(f"  No valid method data found for this configuration, skipping plot generation")
                plt.close()
                continue
            
            if include_baselines:
                filename = f'{self.log_base_path}/NBFL/logs/comparison_{logger_concerning}_mal_{mal}_attack_{attack_type}_alpha_{alpha}_{data_dist}_rewind_{rewind}_nsamples_{nsamples}.png'
            else:
                filename = f'{self.log_base_path}/NBFL/logs/avg_{logger_concerning}_mal_{mal}_attack_{attack_type}_alpha_{alpha}_{data_dist}_rewind_{rewind}_nsamples_{nsamples}.png'
            
            plt.title(title)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filename}")
            plt.clf()
            plt.close()
            
            # Generate legitimate device plots if requested and applicable
            if legitimate_plots and 'local' in logger_concerning and attack_type != 0 and mal > 0:
                plt.figure(figsize=(10, 6))
                
                for i, method in enumerate(methods):
                    logs_for_method = all_method_logs.get(method, [])
                    
                    if not logs_for_method:
                        continue
                    
                    legitimate_avg_values = self.process_logs_for_config(
                        logs_for_method, logger_concerning, legitimate_only=True, mal_count=mal
                    )
                    
                    if legitimate_avg_values:
                        legit_mean_line = np.mean(legitimate_avg_values, axis=0)
                        legit_std = np.std(legitimate_avg_values, axis=0)
                        
                        plt.fill_between(range(1, len(legit_mean_line) + 1), legit_mean_line - legit_std, legit_mean_line + legit_std,
                                        facecolor=colors[i], color=colors[i], alpha=0.2)
                        
                        display_name = method_display_names.get(method, method)
                        plot_label = f'{display_name} ({len(legitimate_avg_values)} seeds)'
                        plt.plot(range(1, len(legit_mean_line) + 1), legit_mean_line, color=colors[i], label=plot_label)
                        
                        # Add text annotations
                        x = 0
                        plt.text(x + 1, legit_mean_line[x], f'{legit_mean_line[x]:.2f}', 
                                ha='center', va='bottom', fontsize=8, color='black')
                        for x in range(4, len(legit_mean_line), 5):
                            plt.text(x + 1, legit_mean_line[x], f'{legit_mean_line[x]:.2f}', 
                                    ha='center', va='bottom', fontsize=8, color='black')
                
                plt.legend(loc='best')
                plt.xlabel('Communication Round')
                plt.ylabel(y_axis_label)
                
                legit_title = f'LEGIT {" ".join(logger_concerning.split("_")).title()} - {mal} Atkers - {self.attack_type_map[attack_type]}, α: {alpha}, {data_dist.upper()}, rewind: {rewind}, n_samples: {nsamples}'
                if include_baselines:
                    legit_filename = f'{self.log_base_path}/NBFL/logs/comparison_{logger_concerning}_mal_{mal}_attack_{attack_type}_alpha_{alpha}_{data_dist}_rewind_{rewind}_nsamples_{nsamples}_legitimate.png'
                else:
                    legit_filename = f'{self.log_base_path}/NBFL/logs/avg_{logger_concerning}_mal_{mal}_attack_{attack_type}_alpha_{alpha}_{data_dist}_rewind_{rewind}_nsamples_{nsamples}_legitimate.png'
                
                plt.title(legit_title)
                plt.savefig(legit_filename, dpi=300, bbox_inches='tight')
                print(f"Saved legitimate plot: {legit_filename}")
                plt.clf()
                plt.close()

    def process_stake_data(self, logger):
        """Process stake data from pos_book logger"""
        avg_stake_over_rounds = []
        for comm_round, device_topos_book in logger['pos_book'].items():
            stake_over_devices = []
            for device_idx, pos_book in device_topos_book.items():
                stake_over_devices.append(list(pos_book.values()))
            stake_over_devices = list(zip(*stake_over_devices))
            avg_stake_over_rounds.append(np.mean(stake_over_devices, axis=1))
        avg_stake_over_rounds = list(zip(*avg_stake_over_rounds))
        return avg_stake_over_rounds

    def generate_stake_plots(self, alpha_filter=None, verbose=True):
        """Generate stake plots across all configurations"""
        if verbose:
            print("Generating stake plots...")
        
        config_groups = self.group_logs_by_config()
        
        # Auto-detect alpha values if not specified
        if alpha_filter is None:
            alpha_filter = self.get_available_alpha_values(config_groups)
            if verbose:
                print(f"Auto-detected alpha values: {alpha_filter}")
        elif verbose:
            print(f"Using specified alpha values: {alpha_filter}")
        
        unique_configs = self.get_unique_configs(config_groups)
        
        for mal, attack_type, alpha, data_dist, rewind, ndevices, nsamples, rounds in unique_configs:
            if self.should_skip_config(mal, attack_type):
                continue
            
            if alpha not in alpha_filter:
                continue
            
            if verbose:
                print(f"Processing stake: mal={mal}, attack={attack_type}, alpha={alpha}, data_dist={data_dist}, rewind={rewind}, ndevices={ndevices}, nsamples={nsamples}, rounds={rounds}")
            
            target_config_key = f"NBFL_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack_type}_rewind_{rewind}"
            
            if target_config_key not in config_groups:
                continue
            
            logs_for_config = config_groups[target_config_key]
            
            # Process stake data across all runs
            stake_data_over_runs = []
            for log_info in logs_for_config:
                logger = self.load_logger_data(log_info['path'], 'pos_book')
                if logger and 'pos_book' in logger:
                    stake_data = self.process_stake_data(logger)
                    stake_data_over_runs.append(stake_data)
            
            if not stake_data_over_runs:
                continue
            
            # Calculate mean across runs
            mean_lines = np.mean(stake_data_over_runs, axis=0)
            
            # Create stake plot
            plt.figure(figsize=(12, 8))
            
            for i, ml in enumerate(mean_lines):
                color = 'green'  # legitimate devices
                if i + 1 + mal > ndevices:  # malicious devices
                    color = 'red'
                    if attack_type == 3 or (attack_type == 4 and (i + 1) % 2 == 0):
                        color = 'magenta'
                
                plt.plot(range(1, len(ml) + 1), ml, color=color)
                plt.annotate(f'{i + 1}', xy=(len(ml), ml[-1]), xytext=(5, 0), 
                           textcoords='offset points', color=color, fontsize=8)
            
            # Add legend
            import matplotlib.lines as mlines
            green_line = mlines.Line2D([], [], color='green', label="legitimate")
            
            legend_handles = [green_line]
            if attack_type in [1, 4]:
                red_line = mlines.Line2D([], [], color='red', label="poison attack")
                legend_handles.append(red_line)
            if attack_type in [3, 4]:
                magenta_line = mlines.Line2D([], [], color='magenta', label="laziness attack")
                legend_handles.append(magenta_line)
            
            plt.legend(handles=legend_handles, loc='best', prop={'size': 8})
            
            plt.xlabel('Communication Round', fontsize=10)
            plt.ylabel('Stake', fontsize=10)
            plt.title(f'Stake Curves - {mal} Atkers - {self.attack_type_map[attack_type]}, α: {alpha}, {data_dist.upper()}, rewind: {rewind}, n_samples: {nsamples}', fontsize=12)
            plt.grid(axis='x')
            plt.xticks(range(1, len(mean_lines[0]) + 1))
            
            filename = f'{self.log_base_path}/NBFL/logs/avg_stake_mal_{mal}_attack_{attack_type}_alpha_{alpha}_{data_dist}_rewind_{rewind}_nsamples_{nsamples}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved stake plot: {filename}")
            plt.clf()
            plt.close()

    def generate_event_plots(self, event_type='forking_event', alpha_filter=None, verbose=True):
        """Generate event plots (forking events or malicious winning count)"""
        if verbose:
            print(f"Generating {event_type} plots...")
        
        config_groups = self.group_logs_by_config()
        
        # Auto-detect alpha values if not specified
        if alpha_filter is None:
            alpha_filter = self.get_available_alpha_values(config_groups)
            if verbose:
                print(f"Auto-detected alpha values: {alpha_filter}")
        elif verbose:
            print(f"Using specified alpha values: {alpha_filter}")
        
        unique_configs = self.get_unique_configs(config_groups)
        
        # Import Rectangle at the top to avoid scoping issues
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        
        # Single plot approach - no alignment issues!
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Build data in the same order as the plot will display
        plot_data = []  # Store all data to ensure consistent ordering
        
        for mal, attack_type, alpha, data_dist, rewind, ndevices, nsamples, rounds in sorted(unique_configs):
            if self.should_skip_config(mal, attack_type):
                continue
            
            if alpha not in alpha_filter:
                continue
            
            target_config_key = f"NBFL_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack_type}_rewind_{rewind}"
            
            if target_config_key not in config_groups:
                continue
            
            logs_for_config = sorted(config_groups[target_config_key], 
                                   key=lambda x: x['seed'], reverse=True)
            
            for log_info in logs_for_config:
                logger = self.load_logger_data(log_info['path'], event_type)
                malicious_rounds = 0
                total_rounds = 0
                winning_events = {}  # Store the actual events for plotting
                
                if logger and event_type in logger:
                    total_rounds = len(logger[event_type])
                    winning_events = logger[event_type]
                    
                    # Count rounds with malicious wins
                    for comm_round, value in logger[event_type].items():
                        if event_type == 'malicious_winning_count' and value > 0:
                            malicious_rounds += 1
                
                # Store data for this row
                label = f'M{mal} - {self.attack_type_map[attack_type]}, sd: {log_info["seed"]}, α: {alpha}, {data_dist}, rewind: {rewind}, n_samples: {nsamples}'
                percentage = (malicious_rounds / total_rounds * 100) if total_rounds > 0 else 0
                
                plot_data.append({
                    'label': label,
                    'events': winning_events,
                    'malicious_rounds': malicious_rounds,
                    'total_rounds': total_rounds,
                    'percentage': percentage
                })
        
        # Calculate plot dimensions
        num_rows = len(plot_data)
        
        if event_type == 'malicious_winning_count':
            # For malicious winning count: include space for statistics columns
            max_round = 25  # Assuming 25 rounds based on your data
            main_plot_width = max_round + 2  # Add some padding
            text_area_start = main_plot_width + 1
            text_area_width = 8  # Space for text columns
            total_width = text_area_start + text_area_width
        else:
            # For forking events: just the main plot area
            max_round = 25
            main_plot_width = max_round + 2
            total_width = main_plot_width
        
        # Set up the plot area
        ax.set_xlim(0, total_width)
        ax.set_ylim(-0.5, num_rows - 0.5)
        
        # Use original data order (no reversal) so M10 appears at top, M0 at bottom
        y_axis_labels = []
        
        # Plot events and add statistics text in one go
        for row_idx, data in enumerate(plot_data):
            # Plot the events in the main area
            for comm_round, value in data['events'].items():
                if event_type == 'forking_event':
                    if value > 1:
                        ax.scatter(comm_round, row_idx, marker='o', color='red', s=50)
                        ax.text(comm_round, row_idx, str(value), ha='center', va='center', color='black', fontsize=8)
                    else:
                        # Make white dots completely invisible - just for x-axis spacing
                        ax.scatter(comm_round, row_idx, marker='o', color='none', s=50, alpha=0)
                elif event_type == 'malicious_winning_count':
                    if value > 0:
                        ax.scatter(comm_round, row_idx, marker='o', color='red', s=50)
                        ax.text(comm_round, row_idx, str(value), ha='center', va='center', color='black', fontsize=8)
            
            y_axis_labels.append(data['label'])
            
            # Add statistics text ONLY for malicious winning count
            if event_type == 'malicious_winning_count':
                # Add background rectangles for the text area
                bg_color = '#F2F2F2' if row_idx % 2 == 0 else 'white'
                if row_idx % 2 == 0:
                    rect = Rectangle((text_area_start - 0.5, row_idx - 0.4), text_area_width + 1, 0.8,
                                   facecolor=bg_color, alpha=0.6, edgecolor='none')
                    ax.add_patch(rect)
                
                # Add the statistics text
                mal_rounds_text = f"{data['malicious_rounds']}/{data['total_rounds']}"
                percentage_text = f"{data['percentage']:.1f}%"
                
                ax.text(text_area_start + 2, row_idx, mal_rounds_text, ha='center', va='center', 
                       fontsize=10, weight='normal')
                ax.text(text_area_start + 5, row_idx, percentage_text, ha='center', va='center', 
                       fontsize=10, weight='normal')
        
        # Add borders and formatting based on plot type
        if event_type == 'malicious_winning_count':
            # Add headers for the statistics columns
            header_y = num_rows - 0.2
            ax.text(text_area_start + 2, header_y, 'Mal Rounds', ha='center', va='center',
                   fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#4472C4', edgecolor='none', alpha=0.8),
                   color='white')
            ax.text(text_area_start + 5, header_y, 'Percentage', ha='center', va='center',
                   fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#4472C4', edgecolor='none', alpha=0.8),
                   color='white')
            
            # Add vertical separator line
            ax.axvline(x=text_area_start + 3.5, color='lightgray', linestyle='-', linewidth=1, alpha=0.7)
            
            # Add border around statistics area
            stats_border = Rectangle((text_area_start - 0.5, -0.45), text_area_width + 1, num_rows + 0.4,
                                   facecolor='none', edgecolor='gray', linewidth=1, alpha=0.5)
            ax.add_patch(stats_border)
            
            # Add main plot border (for malicious winning count with stats)
            main_plot_border = Rectangle((0, -0.45), main_plot_width, num_rows + 0.4,
                                       facecolor='none', edgecolor='gray', linewidth=1, alpha=0.5)
            ax.add_patch(main_plot_border)
        else:
            # For forking events: just add a simple border around the entire plot
            plot_border = Rectangle((0, -0.45), main_plot_width, num_rows + 0.4,
                                  facecolor='none', edgecolor='gray', linewidth=1, alpha=0.5)
            ax.add_patch(plot_border)
        
        # Set up axes and labels
        if event_type == 'forking_event':
            legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Number of Forking Events')]
            title = 'Forking Events'
            legend_location = 'lower right'
        else:
            legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Number of Malicious Winning Blocks')]
            title = 'Malicious Winning Count'
            legend_location = 'lower left'
        
        ax.legend(handles=legend_handles, loc=legend_location, prop={'size': 10})
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Run Name')
        ax.set_title(title)
        ax.set_yticks(range(len(y_axis_labels)))
        ax.set_yticklabels(y_axis_labels)
        
        # Set x-axis ticks only for the main plot area
        ax.set_xticks(range(0, main_plot_width, 5))
        
        plt.tight_layout()
        
        filename = f'{self.log_base_path}/NBFL/logs/{event_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved {event_type} plot: {filename}")
        plt.clf()
        plt.close()

    def generate_winning_validator_plots(self, alpha_filter=None, verbose=True):
        """Generate winning validator plots showing device choices across communication rounds"""
        if verbose:
            print("Generating winning validator plots...")
        
        config_groups = self.group_logs_by_config()
        
        # Auto-detect alpha values if not specified
        if alpha_filter is None:
            alpha_filter = self.get_available_alpha_values(config_groups)
            if verbose:
                print(f"Auto-detected alpha values: {alpha_filter}")
        elif verbose:
            print(f"Using specified alpha values: {alpha_filter}")
        
        unique_configs = self.get_unique_configs(config_groups)
        
        # Import required modules
        from collections import Counter
        import matplotlib.pyplot as plt
        import numpy as np
        
        for mal, attack_type, alpha, data_dist, rewind, ndevices, nsamples, rounds in unique_configs:
            if self.should_skip_config(mal, attack_type):
                continue
            
            if alpha not in alpha_filter:
                continue
            
            if verbose:
                print(f"Processing winning validators: mal={mal}, attack={attack_type}, alpha={alpha}, data_dist={data_dist}, rewind={rewind}, ndevices={ndevices}, nsamples={nsamples}, rounds={rounds}")
            
            target_config_key = f"NBFL_mnist_{data_dist}_alpha_{alpha}_ndevices_{ndevices}_nsamples_{nsamples}_rounds_{rounds}_mal_{mal}_attack_{attack_type}_rewind_{rewind}"
            
            if target_config_key not in config_groups:
                continue
            
            logs_for_config = config_groups[target_config_key]
            
            if not logs_for_config:
                continue
            
            # Process each seed separately
            for log_info in logs_for_config:
                logger = self.load_logger_data(log_info['path'], 'picked_winning_block')
                
                if not logger or 'picked_winning_block' not in logger:
                    if verbose:
                        print(f"  Warning: picked_winning_block not found in {log_info['path']}")
                    continue
                
                seed = log_info['seed']
                
                # Process the winning validator data
                winning_data = logger['picked_winning_block']
                
                # Collect all data points for plotting
                comm_rounds = []
                validator_ids = []
                device_counts = []
                
                for comm_round, device_choices in winning_data.items():
                    # Count how many devices picked each validator
                    validator_counts = Counter(device_choices.values())
                    
                    for validator_id, count in validator_counts.items():
                        comm_rounds.append(comm_round)
                        validator_ids.append(validator_id)
                        device_counts.append(count)
                
                if not comm_rounds:  # Skip if no data
                    if verbose:
                        print(f"  Warning: No winning validator data found for {log_info['path']}")
                    continue
                
                # Create the plot
                fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                
                # Create scatter plot with uniform circle sizes
                circle_size = 200  # Fixed size for all circles
                scatter = ax.scatter(comm_rounds, validator_ids, s=circle_size, 
                                   c='lightblue', edgecolors='black', linewidth=1, alpha=0.8)
                
                # Add numbers inside the circles
                for round_val, validator_id, count in zip(comm_rounds, validator_ids, device_counts):
                    ax.text(round_val, validator_id, str(count), ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='black')
                
                # Customize the plot
                ax.set_xlabel('Communication Round', fontsize=10)
                ax.set_ylabel('Winning Validator ID', fontsize=10)
                ax.set_title(f'Winning Validator Selection - M{mal} - {self.attack_type_map[attack_type]}, seed: {seed}, α: {alpha}, {data_dist.upper()}, n_samples: {nsamples}', fontsize=12)
                
                # Set integer ticks for both axes
                if comm_rounds:
                    min_round = min(comm_rounds)
                    max_round = max(comm_rounds)
                    ax.set_xticks(range(min_round, max_round + 1))
                    ax.set_xlim(min_round - 0.5, max_round + 0.5)
                
                if validator_ids:
                    # Set y-ticks to show all validator IDs that appear
                    unique_validators = sorted(set(validator_ids))
                    ax.set_yticks(unique_validators)
                    ax.set_ylim(min(unique_validators) - 0.5, max(unique_validators) + 0.5)
                
                # Add grid lines for every round and every validator (like stake curves)
                # Vertical grid lines for each communication round
                for round_val in range(min(comm_rounds), max(comm_rounds) + 1):
                    ax.axvline(x=round_val, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
                
                # Horizontal grid lines for each validator ID
                unique_validators = sorted(set(validator_ids))
                for validator_id in unique_validators:
                    ax.axhline(y=validator_id, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
                
                # Save the plot
                filename = f'{self.log_base_path}/NBFL/logs/winning_validator_mal_{mal}_attack_{attack_type}_alpha_{alpha}_{data_dist}_rewind_{rewind}_nsamples_{nsamples}_seed_{seed}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"  Saved: {filename}")
                
                plt.clf()
                plt.close()

# Example usage functions for backward compatibility
def generate_all_plots(log_base_path='/Users/chenhang/Documents/Working', alpha_filter=None, verbose=True):
    """Generate all standard plots with auto-detected alpha values"""
    analyzer = NBFLLogAnalyzer(log_base_path)
    
    # Auto-detect alpha values if not specified
    if alpha_filter is None:
        alpha_filter = analyzer.get_available_alpha_values()
        if verbose:
            print(f"Auto-detected alpha values: {alpha_filter}")
    
    # Display configuration summary
    if verbose:
        summary = analyzer.get_available_configs_summary()
        print("\nConfiguration Summary:")
        print(f"  Alpha values: {summary['alpha_values']}")
        print(f"  Data distributions: {summary['data_distributions']}")
        print(f"  Malicious counts: {summary['mal_values']}")
        print(f"  Attack types: {summary['attack_types']}")
        print(f"  ndevices values: {summary['ndevices_values']}")
        print(f"  nsamples values: {summary['nsamples_values']}")
        print(f"  rounds values: {summary['rounds_values']}")
        print(f"  Total unique configs: {summary['total_configs']}")
        print(f"  Total log groups: {summary['total_log_groups']}")
        print()
    
    # Generate metric plots
    analyzer.generate_metric_plots('global_test_acc', 'Accuracy', alpha_filter, verbose=verbose)
    analyzer.generate_metric_plots('global_model_sparsity', 'Sparsity', alpha_filter, verbose=verbose)
    analyzer.generate_metric_plots('local_max_acc', 'Accuracy', alpha_filter, legitimate_plots=True, verbose=verbose)
    
    # Generate stake plots
    analyzer.generate_stake_plots(alpha_filter, verbose=verbose)
    
    # Generate event plots
    analyzer.generate_event_plots('forking_event', alpha_filter, verbose=verbose)
    analyzer.generate_event_plots('malicious_winning_count', alpha_filter, verbose=verbose)
    
    # Generate winning validator plots
    analyzer.generate_winning_validator_plots(alpha_filter, verbose=verbose)