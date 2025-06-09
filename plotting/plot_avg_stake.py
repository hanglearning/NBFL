"""
Updated plot_avg_stake.py using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def main():
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working')
    
    # Generate stake plots
    # Alpha values are auto-detected from available log folders
    analyzer.generate_stake_plots()
    
    print("Stake plots generated successfully!")

if __name__ == "__main__":
    main()