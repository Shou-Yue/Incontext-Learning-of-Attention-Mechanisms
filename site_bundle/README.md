# Plotly Bundle for GitHub Pages (Option B)

This bundle contains the **results JSONs** and the **Plotly builder script** so you can rebuild the interactive HTML locally (or via GitHub Actions) and then commit the generated HTML to your GitHub Pages repo.

## Contents
- `plot_interactive_experiments.py` — builds `interactive_plots.html`
- `results/exp_steps_sweep/all_results.json`
- `results/exp_layers_sweep/all_results.json`
- `results/exp_context_sweep/all_results.json`
- `results/exp_context_sweep_zero/all_results.json`
- `requirements_plot.txt` — minimal deps for the plotting script

## Prerequisites
- Python 3.10+
- `pip install -r requirements_plot.txt`

## Build the HTML
From the repository where you copy this bundle:

```bash
python3 plot_interactive_experiments.py \
  --steps_results results/exp_steps_sweep/all_results.json \
  --layers_results results/exp_layers_sweep/all_results.json \
  --context_results results/exp_context_sweep/all_results.json \
  --context_zero_results results/exp_context_sweep_zero/all_results.json \
  --output_dir docs/plots
```

This generates:
- `docs/plots/interactive_plots.html`

## Embed in the GitHub Pages site
Add a link in the **Results** section (or wherever you want the graphs) to:

```
plots/interactive_plots.html
```

Example:
```html
<a href="plots/interactive_plots.html">Interactive Experiment Plots</a>
```

Note: GitHub Pages is static, so the Python script must be run **locally** or in CI. The site only serves the generated HTML.
