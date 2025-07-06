"""
Master script to generate all plots using centralized utilities
Updated to work with any available method logs (not just NBFL)
"""

from plotting_utils import NBFLLogAnalyzer

def main():
    """Generate all plots with consistent logic for any available methods"""
    
    # Configuration
    log_base_path = '/Users/chenhang/Documents/Working'
    
    print("Starting comprehensive plot generation...")
    print(f"Base path: {log_base_path}")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path)
    
    # Display available configurations
    summary = analyzer.get_available_configs_summary()
    print("\nDetected Configuration Summary:")
    print(f"  Alpha values: {summary['alpha_values']}")
    print(f"  Data distributions: {summary['data_distributions']}")
    print(f"  Malicious counts: {summary['mal_values']}")
    print(f"  Attack types: {summary['attack_types']}")
    print(f"  ndevices values: {summary['ndevices_values']}")
    print(f"  nsamples values: {summary['nsamples_values']}")
    print(f"  rounds values: {summary['rounds_values']}")
    print(f"  Total unique configs: {summary['total_configs']}")
    print(f"  Total log groups: {summary['total_log_groups']}")
    
    # Show what methods are available
    config_groups = analyzer.group_logs_by_config()
    available_methods = set()
    for config_key in config_groups.keys():
        method = config_key.split('_')[0]
        available_methods.add(method)
    print(f"  Available methods: {sorted(available_methods)}")
    print("=" * 60)
    
    # 1. Generate accuracy plots with all available methods
    print("\n1. Generating accuracy plots with all available methods...")
    analyzer.generate_metric_plots(
        logger_concerning='global_test_acc',
        y_axis_label='Accuracy',
        legitimate_plots=False,
        include_baselines=True
    )
    
    analyzer.generate_metric_plots(
        logger_concerning='local_max_acc',
        y_axis_label='Accuracy',
        legitimate_plots=True,
        include_baselines=True
    )
    
    # 2. Generate sparsity plots with all available methods
    print("\n2. Generating sparsity plots with all available methods...")
    analyzer.generate_metric_plots(
        logger_concerning='global_model_sparsity',
        y_axis_label='Sparsity',
        legitimate_plots=False,
        include_baselines=True
    )
    
    # 3. Generate stake plots (only for NBFL method)
    print("\n3. Generating stake plots...")
    if 'NBFL' in available_methods:
        analyzer.generate_stake_plots()
    else:
        print("   Skipping stake plots (requires NBFL logs)")
    
    # 4. Generate event plots (only for NBFL method)
    print("\n4. Generating event plots...")
    if 'NBFL' in available_methods:
        analyzer.generate_event_plots(event_type='forking_event')
        analyzer.generate_event_plots(event_type='malicious_winning_count')
    else:
        print("   Skipping event plots (requires NBFL logs)")
    
    # 5. Generate winning validator plots (only for NBFL method)
    print("\n5. Generating winning validator plots...")
    if 'NBFL' in available_methods:
        analyzer.generate_winning_validator_plots()
    else:
        print("   Skipping winning validator plots (requires NBFL logs)")
    
    print("\n" + "=" * 60)
    print("All available plots generated successfully!")
    print("Check the logs directory for output files.")

if __name__ == "__main__":
    main()