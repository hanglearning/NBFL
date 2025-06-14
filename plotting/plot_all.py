"""
Master script to generate all plots using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def main():
    """Generate all plots with consistent logic"""
    
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
    print(f"  Total unique configs: {summary['total_configs']}")
    print(f"  Total log groups: {summary['total_log_groups']}")
    print("=" * 60)
    
    # 1. Generate accuracy plots
    print("\n1. Generating accuracy plots...")
    analyzer.generate_metric_plots(
        logger_concerning='global_test_acc',
        y_axis_label='Accuracy',
        legitimate_plots=False
    )
    
    analyzer.generate_metric_plots(
        logger_concerning='local_test_acc',
        y_axis_label='Accuracy',
        legitimate_plots=True
    )
    
    analyzer.generate_metric_plots(
        logger_concerning='local_max_acc',
        y_axis_label='Accuracy',
        legitimate_plots=True
    )
    
    # 2. Generate sparsity plots
    print("\n2. Generating sparsity plots...")
    analyzer.generate_metric_plots(
        logger_concerning='global_model_sparsity',
        y_axis_label='Sparsity',
        legitimate_plots=False
    )
    
    # 3. Generate stake plots
    print("\n3. Generating stake plots...")
    analyzer.generate_stake_plots()
    
    # 4. Generate event plots
    print("\n4. Generating event plots...")
    analyzer.generate_event_plots(event_type='forking_event')
    analyzer.generate_event_plots(event_type='malicious_winning_count')
    
    # 5. Generate winning validator plots
    print("\n5. Generating winning validator plots...")
    analyzer.generate_plot_validator_selections_plots()
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("Check the logs directory for output files.")

if __name__ == "__main__":
    main()