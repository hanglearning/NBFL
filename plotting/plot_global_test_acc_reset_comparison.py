"""
Script to compare reset 0 vs reset 1 for global test accuracy
"""

from plotting_utils_reset_comparison import NBFLLogAnalyzer

def main():
    # Initialize analyzer with your log base path
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working/NBFL/logs')
    
    print("="*60)
    print("Generating Global Test Accuracy Comparison (Reset 0 vs Reset 1)")
    print("="*60)
    
    # Generate global test accuracy plots comparing reset 0 vs reset 1
    analyzer.generate_metric_plots_with_reset_comparison(
        logger_concerning='global_test_acc',
        y_axis_label='Accuracy',
        legitimate_plots=False,  # Global test acc doesn't need legitimate-only plots
        include_baselines=False,  # Set to True if you want baselines too
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Global test accuracy reset comparison plots generated successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
