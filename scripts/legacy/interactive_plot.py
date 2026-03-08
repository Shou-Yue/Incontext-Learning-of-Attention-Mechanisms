#!/usr/bin/env python3
"""
Interactive visualization of evaluation results with adjustable smoothing and filtering.
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from scipy.ndimage import gaussian_filter1d

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def smooth_curve(data, sigma):
    """Apply Gaussian smoothing to data."""
    if sigma == 0:
        return data
    return gaussian_filter1d(data, sigma=sigma)

class InteractivePlot:
    def __init__(self, results):
        self.results = results
        self.n_points_list = results['n_points_list']
        self.n_dims = results['n_dims']
        
        # Create figure with space for controls
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = plt.subplot2grid((6, 3), (0, 0), colspan=3, rowspan=4)
        
        # Initial settings
        self.smoothing = 0
        self.show_bands = True
        self.x_range = 'all'
        self.y_scale = 'linear'
        self.visible_methods = {
            'transformer': True,
            'least_squares': True,
            'knn': True,
            'averaging': True
        }
        
        # Setup controls
        self.setup_controls()
        
        # Initial plot
        self.update_plot()
        
    def setup_controls(self):
        """Setup interactive controls."""
        # Smoothing slider
        ax_smooth = plt.subplot2grid((6, 3), (4, 0), colspan=3)
        self.slider_smooth = Slider(
            ax_smooth, 'Smoothing', 0, 5, valinit=0, valstep=0.5
        )
        self.slider_smooth.on_changed(self.on_smooth_change)
        
        # Checkboxes for error bands
        ax_bands = plt.subplot2grid((6, 3), (5, 0))
        self.check_bands = CheckButtons(ax_bands, ['Show Error Bands'], [True])
        self.check_bands.on_clicked(self.on_bands_toggle)
        
        # Radio buttons for x-range
        ax_xrange = plt.subplot2grid((6, 3), (5, 1))
        self.radio_xrange = RadioButtons(
            ax_xrange, 
            ('All', 'Underparam (n<d)', 'Overparam (n≥d)'),
            active=0
        )
        self.radio_xrange.on_clicked(self.on_xrange_change)
        
        # Radio buttons for y-scale
        ax_yscale = plt.subplot2grid((6, 3), (5, 2))
        self.radio_yscale = RadioButtons(
            ax_yscale,
            ('Linear', 'Log'),
            active=0
        )
        self.radio_yscale.on_clicked(self.on_yscale_change)
        
    def on_smooth_change(self, val):
        """Handle smoothing slider change."""
        self.smoothing = val
        self.update_plot()
        
    def on_bands_toggle(self, label):
        """Handle error bands checkbox toggle."""
        self.show_bands = not self.show_bands
        self.update_plot()
        
    def on_xrange_change(self, label):
        """Handle x-range radio button change."""
        if label == 'All':
            self.x_range = 'all'
        elif label == 'Underparam (n<d)':
            self.x_range = 'under'
        else:  # Overparam (n≥d)
            self.x_range = 'over'
        self.update_plot()
        
    def on_yscale_change(self, label):
        """Handle y-scale radio button change."""
        self.y_scale = label.lower()
        self.update_plot()
        
    def filter_data_by_xrange(self):
        """Filter data based on x-range selection."""
        n_points = np.array(self.n_points_list)
        
        if self.x_range == 'all':
            mask = np.ones(len(n_points), dtype=bool)
        elif self.x_range == 'over':
            mask = n_points >= self.n_dims
        else:  # under
            mask = n_points < self.n_dims
            
        return mask
        
    def update_plot(self):
        """Update the plot with current settings."""
        self.ax.clear()
        
        # Get data filtering mask
        mask = self.filter_data_by_xrange()
        n_points_filtered = np.array(self.n_points_list)[mask]
        
        if len(n_points_filtered) == 0:
            self.ax.text(0.5, 0.5, 'No data in selected range', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return
        
        # Colors
        colors = {
            'transformer': '#4472C4',
            'least_squares': '#ED7D31',
            'knn': '#70AD47',
            'averaging': '#C55A11'
        }
        
        # Labels and markers
        methods_info = {
            'transformer': ('Transformer', 'o-'),
            'least_squares': ('Least Squares', 's-'),
            'knn': ('3-Nearest Neighbors', '^-'),
            'averaging': ('Averaging', 'd-')
        }
        
        # Plot each method
        for method_key, (label, marker) in methods_info.items():
            if method_key not in self.results:
                continue
                
            mean_vals = np.array(self.results[method_key]['mean'])[mask]
            std_vals = np.array(self.results[method_key]['std'])[mask]
            
            # Filter out NaN
            valid = ~np.isnan(mean_vals)
            if not valid.any():
                continue
                
            x = n_points_filtered[valid]
            mean = mean_vals[valid]
            std = std_vals[valid]
            
            # Apply smoothing
            if self.smoothing > 0 and len(mean) > 3:
                mean = smooth_curve(mean, self.smoothing)
                std = smooth_curve(std, self.smoothing)
            
            # Plot line
            self.ax.plot(x, mean, marker, label=label, 
                        linewidth=2, markersize=5, color=colors[method_key])
            
            # Plot error bands
            if self.show_bands:
                self.ax.fill_between(
                    x,
                    np.maximum(0, mean - std),
                    mean + std,
                    alpha=0.15,
                    color=colors[method_key]
                )
        
        # Add reference lines
        if self.x_range == 'all':
            self.ax.axvline(x=self.n_dims, color='gray', linestyle=':', 
                          alpha=0.7, linewidth=1.5, label=f'n=d={self.n_dims}')
        
        self.ax.axhline(y=1.0, color='gray', linestyle='--', 
                       alpha=0.4, linewidth=1)
        
        # Formatting
        self.ax.set_xlabel('Number of In-Context Examples', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Squared Error', fontsize=12, fontweight='bold')
        
        title = f'Linear Functions ({self.n_dims}D)'
        if self.x_range == 'over':
            title += ' - Overparameterized Regime'
        elif self.x_range == 'under':
            title += ' - Underparameterized Regime'
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Set y-scale
        if self.y_scale == 'log':
            self.ax.set_yscale('log')
            self.ax.set_ylim(bottom=1e-4)
        else:
            self.ax.set_yscale('linear')
            # Adjust ylim based on visible data
            all_means = []
            for method_key in methods_info.keys():
                if method_key in self.results:
                    means = np.array(self.results[method_key]['mean'])[mask]
                    valid = ~np.isnan(means)
                    if valid.any():
                        all_means.extend(means[valid])
            if all_means:
                max_val = np.percentile(all_means, 95)  # Use 95th percentile
                self.ax.set_ylim(0, max_val * 1.2)
        
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=10, loc='best', framealpha=0.95)
        
        # Add info text
        info_text = f'Smoothing: {self.smoothing:.1f}'
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        self.fig.canvas.draw_idle()

def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_plot.py <results.json>")
        print("\nExample:")
        print("  python interactive_plot.py results/all_methods_results.json")
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    try:
        results = load_results(results_path)
        print(f"Loaded results from {results_path}")
        print(f"Dimension: {results['n_dims']}")
        print(f"Context lengths: {min(results['n_points_list'])} to {max(results['n_points_list'])}")
        print("\nInteractive controls:")
        print("  - Smoothing slider: Adjust curve smoothing (0-5)")
        print("  - Show Error Bands: Toggle confidence intervals")
        print("  - X-range buttons: Focus on different regimes")
        print("  - Y-scale buttons: Switch between linear and log scale")
        print("\nClose the plot window to exit.")
        
        plot = InteractivePlot(results)
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File '{results_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file '{results_path}'")
        sys.exit(1)

if __name__ == '__main__':
    main()
