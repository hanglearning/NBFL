"""
Updated plot_avg_global_test_acc.py using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def main():
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working')
    
    # Generate global test accuracy plots with baseline comparisons
    # Alpha values are auto-detected from available log folders
    analyzer.generate_metric_plots(
        logger_concerning='global_test_acc',
        y_axis_label='Accuracy',
        legitimate_plots=False,  # Global test acc doesn't need legitimate-only plots
        include_baselines=True   # Include Standalone, FedAvg, LotteryFL, PoIS comparisons
    )
    
    print("Global test accuracy comparison plots generated successfully!")

if __name__ == "__main__":
    main()