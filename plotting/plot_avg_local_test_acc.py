"""
Updated plot_avg_local_test_acc.py using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def main():
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working')
    
    # Generate local test accuracy plots
    # This will generate both all-devices and legitimate-only plots
    # Alpha values are auto-detected from available log folders
    
    analyzer.generate_metric_plots(
        logger_concerning='local_test_acc',
        y_axis_label='Accuracy',
        legitimate_plots=True  # This will generate both regular and legitimate-only plots
    )
    
    print("Local test accuracy plots generated successfully!")

if __name__ == "__main__":
    main()