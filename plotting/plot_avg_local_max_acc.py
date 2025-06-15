"""
Updated plot_avg_local_max_acc.py using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def main():
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working')
    
    # Generate local max accuracy plots with baseline comparisons
    # This will generate both all-devices and legitimate-only plots
    # Alpha values are auto-detected from available log folders
    
    analyzer.generate_metric_plots(
        logger_concerning='local_max_acc',
        y_axis_label='Accuracy',
        legitimate_plots=True,   # This will generate both regular and legitimate-only plots
        include_baselines=True   # Include Standalone, FedAvg, LotteryFL, PoIS comparisons
    )
    
    print("Local max accuracy comparison plots generated successfully!")

if __name__ == "__main__":
    main()