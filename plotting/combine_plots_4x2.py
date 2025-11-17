import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict

def parse_plot_filename(filename):
    """
    Parse plot filename to extract mal count and data distribution
    Works with various plot types (comparison_, avg_stake_, etc.)
    """
    # Generic pattern to match mal_X and iid/non-iid in any plot filename
    pattern = r'.*mal_(\d+).*_(iid|non-iid).*\.png$'
    match = re.search(pattern, filename)
    
    if match:
        mal_count = int(match.group(1))
        data_dist = match.group(2)
        return mal_count, data_dist
    return None, None

def extract_plot_pattern(filename):
    """
    Extract the base plot pattern from filename
    
    Examples:
        comparison_global_test_acc_mal_3_attack_4_alpha_∞_iid_reset_0.png
        -> comparison_global_test_acc
        
        old_vs_new_local_max_acc_mal_6_attack_4_alpha_1.0_non-iid.png
        -> old_vs_new_local_max_acc
        
        reset_comparison_global_test_acc_mal_10_attack_4_alpha_∞_iid_legitimate.png
        -> reset_comparison_global_test_acc
    """
    # Remove .png extension
    name = filename.replace('.png', '')
    
    # Pattern to extract everything before mal_X
    pattern = r'^(.+?)_mal_\d+'
    match = re.match(pattern, name)
    
    if match:
        base_pattern = match.group(1)
        
        # Check if it ends with _legitimate and remove it for grouping
        # We'll handle legitimate separately
        if '_legitimate' in name:
            return base_pattern, True
        return base_pattern, False
    
    return None, False

def discover_plot_patterns(directory, verbose=True):
    """
    Automatically discover all unique plot patterns in a directory
    
    Returns:
        Dictionary mapping pattern names to lists of matching files
    """
    patterns = defaultdict(lambda: {'regular': [], 'legitimate': []})
    
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            pattern, is_legitimate = extract_plot_pattern(filename)
            
            if pattern:
                if is_legitimate:
                    patterns[pattern]['legitimate'].append(filename)
                else:
                    patterns[pattern]['regular'].append(filename)
    
    if verbose:
        print("="*70)
        print("DISCOVERED PLOT PATTERNS")
        print("="*70)
        for pattern, files in sorted(patterns.items()):
            regular_count = len(files['regular'])
            legit_count = len(files['legitimate'])
            print(f"\n📊 {pattern}")
            print(f"   Regular plots: {regular_count}")
            if legit_count > 0:
                print(f"   Legitimate plots: {legit_count}")
            print(f"   Total: {regular_count + legit_count}")
    
    return patterns

def find_matching_plots(directory, plot_pattern, is_legitimate=False):
    """
    Find all plots matching a specific pattern and organize by mal count and distribution
    
    Args:
        directory: Directory containing the plot files
        plot_pattern: Pattern to match (e.g., "comparison_global_test_acc")
        is_legitimate: Whether to look for _legitimate plots
    
    Returns:
        Dictionary organized by mal count and data distribution
    """
    plot_files = {}
    
    for filename in os.listdir(directory):
        if not filename.endswith('.png'):
            continue
        
        # Check if this file matches our pattern
        file_pattern, file_is_legit = extract_plot_pattern(filename)
        
        if file_pattern == plot_pattern and file_is_legit == is_legitimate:
            mal_count, data_dist = parse_plot_filename(filename)
            if mal_count is not None and data_dist is not None:
                if mal_count not in plot_files:
                    plot_files[mal_count] = {}
                plot_files[mal_count][data_dist] = filename
    
    return plot_files

def create_combined_plot(directory, plot_pattern, is_legitimate=False, output_filename=None, figsize=(16, 20)):
    """
    Create a combined plot grid from individual plots
    
    Args:
        directory: Directory containing the plot files
        plot_pattern: Pattern to match plot files (e.g., "comparison_global_test_acc")
        is_legitimate: Whether this is for legitimate-only plots
        output_filename: Name for the combined plot (auto-generated if None)
        figsize: Size of the combined figure
    """
    
    # Find matching plots
    plot_files = find_matching_plots(directory, plot_pattern, is_legitimate)
    
    if not plot_files:
        print(f"No plots found matching pattern: {plot_pattern}" + 
              (" (legitimate)" if is_legitimate else ""))
        return None
    
    # Get sorted mal counts for consistent ordering
    mal_counts = sorted(plot_files.keys())
    num_rows = len(mal_counts)
    num_cols = 2  # iid and non-iid
    
    print(f"\n  Found plots for mal counts: {mal_counts}")
    print(f"  Creating {num_rows} x {num_cols} grid...")
    
    # Create the combined plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Ensure axes is always 2D array for consistent indexing
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Load and display each plot
    for row, mal_count in enumerate(mal_counts):
        for col, data_dist in enumerate(['iid', 'non-iid']):
            ax = axes[row, col]
            
            if data_dist in plot_files[mal_count]:
                filename = plot_files[mal_count][data_dist]
                filepath = os.path.join(directory, filename)
                
                try:
                    img = mpimg.imread(filepath)
                    ax.imshow(img)
                    ax.axis('off')  # Hide axes
                    
                except Exception as e:
                    print(f"    ⚠️  Error loading {filename}: {e}")
                    ax.text(0.5, 0.5, f'Error:\nMal {mal_count}\n{data_dist}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                # Missing plot - show placeholder
                ax.text(0.5, 0.5, f'Missing:\nMal {mal_count}\n{data_dist}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
                ax.axis('off')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Generate output filename if not provided
    if output_filename is None:
        suffix = "_legitimate" if is_legitimate else ""
        output_filename = f"combined_{plot_pattern}{suffix}.png"
    
    # Save the combined plot
    output_path = os.path.join(directory, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_filename}")
    
    plt.close()
    
    return output_path

def combine_all_plots_auto(directory, figsize=(16, 20), skip_patterns=None, verbose=True):
    """
    Automatically discover and combine all plot patterns in a directory
    
    Args:
        directory: Directory containing the plot files
        figsize: Size of each combined figure
        skip_patterns: List of patterns to skip (optional)
        verbose: Print detailed progress
    
    Returns:
        List of created combined plot files
    """
    
    if skip_patterns is None:
        skip_patterns = []
    
    print("="*70)
    print("AUTOMATIC PLOT COMBINER")
    print("="*70)
    print(f"Directory: {directory}")
    print(f"Figure size: {figsize}")
    if skip_patterns:
        print(f"Skipping patterns: {skip_patterns}")
    print("="*70)
    
    # Discover all patterns
    patterns = discover_plot_patterns(directory, verbose=verbose)
    
    if not patterns:
        print("\n❌ No plot patterns found in directory!")
        return []
    
    print(f"\n{'='*70}")
    print(f"CREATING COMBINED PLOTS")
    print(f"{'='*70}")
    
    created_files = []
    
    # Process each pattern
    for pattern in sorted(patterns.keys()):
        if pattern in skip_patterns:
            print(f"\n⏭️  Skipping: {pattern}")
            continue
        
        files = patterns[pattern]
        
        # Process regular plots
        if files['regular']:
            print(f"\n📊 Processing: {pattern}")
            output_file = create_combined_plot(
                directory=directory,
                plot_pattern=pattern,
                is_legitimate=False,
                figsize=figsize
            )
            if output_file:
                created_files.append(output_file)
        
        # Process legitimate plots
        if files['legitimate']:
            print(f"\n📊 Processing: {pattern} (legitimate)")
            output_file = create_combined_plot(
                directory=directory,
                plot_pattern=pattern,
                is_legitimate=True,
                figsize=figsize
            )
            if output_file:
                created_files.append(output_file)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total patterns found: {len(patterns)}")
    print(f"Patterns processed: {len([p for p in patterns.keys() if p not in skip_patterns])}")
    print(f"Combined plots created: {len(created_files)}")
    print(f"{'='*70}")
    
    if created_files:
        print(f"\n✅ Created {len(created_files)} combined plots:")
        for filepath in created_files:
            print(f"   {os.path.basename(filepath)}")
    
    return created_files

def main():
    """
    Main function with automatic plot combination
    """
    
    # ========================================================================
    # CONFIGURATION - UPDATE THIS
    # ========================================================================
    
    # Directory containing your plot files
    directory = "/Users/chenhang/Documents/Working/NBFL/logs"
    
    # Figure size for combined plots (width, height in inches)
    # Adjust based on your needs - larger = higher resolution but bigger files
    figsize = (16, 20)
    
    # Optional: Skip certain patterns if needed
    # Example: skip_patterns = ["avg_stake", "forking_event"]
    skip_patterns = []
    
    # ========================================================================
    
    # Run automatic combination
    combine_all_plots_auto(
        directory=directory,
        figsize=figsize,
        skip_patterns=skip_patterns,
        verbose=True
    )
    
    print("\n🎉 All done!")

# Utility function for easy usage
def combine_plots_by_pattern(directory, pattern, output_name=None, is_legitimate=False, figsize=(16, 20)):
    """
    Manually combine plots for a specific pattern
    
    Usage examples:
        combine_plots_by_pattern("/path/to/plots", "comparison_global_test_acc")
        combine_plots_by_pattern("/path/to/plots", "local_max_acc", is_legitimate=True)
        combine_plots_by_pattern("/path/to/plots", "avg_stake", output_name="my_stakes.png")
    """
    return create_combined_plot(directory, pattern, is_legitimate, output_name, figsize)

if __name__ == "__main__":
    main()