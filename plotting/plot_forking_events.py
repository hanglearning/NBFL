"""
Updated event plotting scripts using centralized utilities
"""

from plotting_utils import NBFLLogAnalyzer

def plot_forking_events():
    """Updated plot_forking_events.py"""
    # Initialize analyzer
    analyzer = NBFLLogAnalyzer(log_base_path='/Users/chenhang/Documents/Working')
    
    # Generate forking events plot
    # Alpha values are auto-detected from available log folders
    analyzer.generate_event_plots(event_type='forking_event')
    
    print("Forking events plot generated successfully!")

def main():
    plot_forking_events()

if __name__ == "__main__":
    main()