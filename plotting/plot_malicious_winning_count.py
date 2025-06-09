"""
Updated event plotting scripts using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def plot_malicious_winning_count():
    """Updated plot_malicious_winning_count.py"""
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working')
    
    # Generate malicious winning count plot
    # Alpha values are auto-detected from available log folders
    analyzer.generate_event_plots(event_type='malicious_winning_count')
    
    print("Malicious winning count plot generated successfully!")

def main():
    plot_malicious_winning_count()

if __name__ == "__main__":
    main()