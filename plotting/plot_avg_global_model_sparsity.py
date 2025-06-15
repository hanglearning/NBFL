"""
Updated plot for global model sparsity using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def main():
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working')
    
    # Generate global model sparsity plots with baseline comparisons
    # Alpha values are auto-detected from available log folders
    analyzer.generate_metric_plots(
        logger_concerning='global_model_sparsity',
        y_axis_label='Sparsity',
        legitimate_plots=False,  # Sparsity doesn't need legitimate-only plots
        include_baselines=True   # Include Standalone, FedAvg, LotteryFL, PoIS comparisons
    )
    
    print("Global model sparsity comparison plots generated successfully!")

if __name__ == "__main__":
    main()