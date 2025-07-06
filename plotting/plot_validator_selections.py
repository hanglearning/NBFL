
"""
Updated plot_validator_selections.py using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def plot_validator_selections():
    """Updated plot_validator_selections.py"""
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working')
    
    # Generate winning validator plots
    # Alpha values are auto-detected from available log folders
    analyzer.generate_winning_validator_plots()
    
    print("Winning validator plots generated successfully!")

def main():
    plot_validator_selections()

if __name__ == "__main__":
    main()