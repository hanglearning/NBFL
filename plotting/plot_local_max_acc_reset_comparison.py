"""
Script to compare reset 0 vs reset 1 for local max accuracy
"""

from plotting_utils_reset_comparison import NBFLLogAnalyzer

def main():
    # Initialize analyzer with your log base path
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working/NBFL/logs')
    
    print("="*60)
    print("Generating Local Max Accuracy Comparison (Reset 0 vs Reset 1)")
    print("="*60)
    
    # Generate local max accuracy plots comparing reset 0 vs reset 1
    # This will generate both all-devices and legitimate-only plots
    analyzer.generate_metric_plots_with_reset_comparison(
        logger_concerning='local_max_acc',
        y_axis_label='Accuracy',
        legitimate_plots=True,  # This will also generate legitimate-only plots
        include_baselines=False,  # Set to True if you want baselines too
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Local max accuracy reset comparison plots generated successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
