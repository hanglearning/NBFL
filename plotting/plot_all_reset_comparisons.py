"""
Comprehensive script to generate all reset 0 vs reset 1 comparison plots
"""

from plotting_utils_reset_comparison import NBFLLogAnalyzer

def main():
    # Initialize analyzer with your log base path
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working/NBFL/logs')
    
    print("="*70)
    print(" "*15 + "NBFL RESET 0 vs RESET 1 COMPARISON")
    print("="*70)
    
    # 1. Generate global test accuracy comparison
    print("\n\n1. Generating Global Test Accuracy Comparison")
    print("-"*70)
    analyzer.generate_metric_plots_with_reset_comparison(
        logger_concerning='global_test_acc',
        y_axis_label='Accuracy',
        legitimate_plots=False,
        include_baselines=False,
        verbose=True
    )
    
    # 2. Generate local max accuracy comparison
    print("\n\n2. Generating Local Max Accuracy Comparison")
    print("-"*70)
    analyzer.generate_metric_plots_with_reset_comparison(
        logger_concerning='local_max_acc',
        y_axis_label='Accuracy',
        legitimate_plots=True,  # Will generate both all-devices and legitimate-only plots
        include_baselines=False,
        verbose=True
    )
    
    # Optional: Generate other metrics if needed
    # Uncomment the following sections if you want to compare reset for other metrics
    
    # 3. Generate global model sparsity comparison (optional)
    # print("\n\n3. Generating Global Model Sparsity Comparison")
    # print("-"*70)
    # analyzer.generate_metric_plots_with_reset_comparison(
    #     logger_concerning='global_model_sparsity',
    #     y_axis_label='Sparsity',
    #     legitimate_plots=False,
    #     include_baselines=False,
    #     verbose=True
    # )
    
    print("\n" + "="*70)
    print("All reset comparison plots generated successfully!")
    print(f"Plots saved to: {analyzer.log_base_path}")
    print("="*70)
    print("\nGenerated plot files:")
    print("  - reset_comparison_global_test_acc_mal_*_attack_*_alpha_*_*.png")
    print("  - reset_comparison_local_max_acc_mal_*_attack_*_alpha_*_*.png")
    print("  - reset_comparison_local_max_acc_mal_*_attack_*_alpha_*_*_legitimate.png")
    print("="*70)

if __name__ == "__main__":
    main()
