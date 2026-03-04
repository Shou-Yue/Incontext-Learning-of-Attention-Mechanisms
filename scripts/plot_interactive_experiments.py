#!/usr/bin/env python3
"""
Interactive plotting with tabs per experiment and sidebar checkboxes.

Outputs a single HTML file with tabs:
  - steps sweep
  - layers sweep
  - context sweep (trained)
  - context sweep (zero training)
Each tab contains two plots: MSE and Cosine Similarity.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import re

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    raise SystemExit("plotly is required: pip install plotly") from e


def _collect_methods(results, lowrank_mode="auto"):
    methods = set()
    for r in results:
        for k in r.keys():
            if k.startswith("mse_") and k.endswith("_mean") and k not in ("mse_gd_mean",):
                method = k.replace("mse_", "").replace("_mean", "")
                if lowrank_mode == "separate":
                    methods.add(method)
                elif method.startswith("lowrank_"):
                    m = re.match(r"lowrank_(?:block\d+_)?k([0-9.]+)", method)
                    if m:
                        ratio = m.group(1)
                        methods.add(f"lowrank_k{ratio}")
                    else:
                        methods.add(method)
                else:
                    methods.add(method)
    return sorted(methods)


def _label(method):
    if method == "lsa":
        return "LSA"
    if method == "softmax":
        return "Softmax"
    if method == "kernel":
        return "Kernelized Linear"
    if method == "gla":
        return "GLA"
    if method == "gqa":
        return "GQA"
    if method == "sparse":
        return "Sparse Causal"
    if method.startswith("lowrank_k"):
        return method.replace("lowrank_", "Low-rank ")
    if method.startswith("lowrank_block"):
        return method.replace("lowrank_", "Low-rank ")
    return method


def _extract_series(results, key_prefix):
    vals = []
    stds = []
    for r in results:
        vals.append(r.get(f"{key_prefix}_mean", np.nan))
        stds.append(r.get(f"{key_prefix}_std", np.nan))
    return np.array(vals, dtype=float), np.array(stds, dtype=float)


def _resolve_method_prefix(r, method_key, lowrank_mode):
    if lowrank_mode == "separate" or not method_key.startswith("lowrank_k"):
        return method_key
    ratio = method_key.replace("lowrank_k", "")
    normal_key = f"lowrank_k{ratio}"
    if f"mse_{normal_key}_mean" in r:
        return normal_key
    # fallback to any block variant with same ratio
    for k in r.keys():
        if k.startswith("mse_lowrank_block") and k.endswith("_mean") and f"_k{ratio}_mean" in k:
            return k[len("mse_"):-len("_mean")]
    return None


def _extract_series_for_method(results, method_key, lowrank_mode):
    vals = []
    stds = []
    for r in results:
        resolved = _resolve_method_prefix(r, method_key, lowrank_mode)
        if resolved is None:
            vals.append(np.nan)
            stds.append(np.nan)
        else:
            vals.append(r.get(f"mse_{resolved}_mean", np.nan))
            stds.append(r.get(f"mse_{resolved}_std", np.nan))
    return np.array(vals, dtype=float), np.array(stds, dtype=float)


def _extract_series_for_method_cos(results, method_key, lowrank_mode):
    vals = []
    stds = []
    for r in results:
        resolved = _resolve_method_prefix(r, method_key, lowrank_mode)
        if resolved is None:
            vals.append(np.nan)
            stds.append(np.nan)
        else:
            vals.append(r.get(f"cosine_sim_{resolved}_mean", np.nan))
            stds.append(r.get(f"cosine_sim_{resolved}_std", np.nan))
    return np.array(vals, dtype=float), np.array(stds, dtype=float)


def _build_fig(results, x_key, x_label, title_prefix, color_map, lowrank_mode):
    results = sorted(results, key=lambda r: r.get(x_key, 0))
    x = np.array([r.get(x_key, 0) for r in results])
    methods = _collect_methods(results, lowrank_mode=lowrank_mode)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{title_prefix} MSE", f"{title_prefix} Cosine Similarity"],
        horizontal_spacing=0.12,
    )

    trace_indices = {}
    label_map = {}

    # Add GD toggle entry (MSE trace only)
    trace_indices["gd"] = [len(fig.data)]
    label_map["gd"] = "T-step GD"

    # MSE baseline (GD)
    mse_gd, mse_gd_std = _extract_series(results, "mse_gd")
    gd_color = color_map.get("gd")
    fig.add_trace(
        go.Scatter(
            x=x, y=mse_gd,
            mode="lines+markers",
            name="T-step GD",
            line=dict(color=gd_color),
            marker=dict(color=gd_color),
            error_y=dict(type="data", array=mse_gd_std, visible=True),
            showlegend=True,
        ),
        row=1, col=1
    )

    # Other methods (MSE + Cosine)
    for m in methods:
        mse_vals, mse_std = _extract_series_for_method(results, m, lowrank_mode)
        cos_vals, cos_std = _extract_series_for_method_cos(results, m, lowrank_mode)
        label = _label(m)
        label_map[m] = label
        trace_indices[m] = []
        color = color_map.get(m)
        fig.add_trace(
            go.Scatter(
                x=x, y=mse_vals,
                mode="lines+markers",
                name=label,
                line=dict(color=color),
                marker=dict(color=color),
                error_y=dict(type="data", array=mse_std, visible=True),
                showlegend=False,
            ),
            row=1, col=1
        )
        trace_indices[m].append(len(fig.data) - 1)
        fig.add_trace(
            go.Scatter(
                x=x, y=cos_vals,
                mode="lines+markers",
                name=f"{label} vs GD",
                line=dict(color=color),
                marker=dict(color=color),
                error_y=dict(type="data", array=cos_std, visible=True),
                showlegend=False,
            ),
            row=1, col=2
        )
        trace_indices[m].append(len(fig.data) - 1)

    # Perfect alignment line
    fig.add_trace(
        go.Scatter(
            x=x, y=np.ones_like(x),
            mode="lines",
            name="Perfect Alignment",
            line=dict(dash="dash", color="red"),
            showlegend=True,
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_xaxes(title_text=x_label, row=1, col=2)
    fig.update_yaxes(title_text="Mean Squared Error", row=1, col=1)
    fig.update_yaxes(title_text="Cosine Similarity", row=1, col=2)
    fig.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, x=0, xanchor="left"),
    )

    return fig, trace_indices, label_map


def _load_if_exists(path_str):
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.exists():
        print(f"[warn] Results file not found: {p}")
        return None
    return json.loads(p.read_text())


def _build_color_map(method_keys):
    # High-contrast qualitative palette (24 colors)
    palette = [
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
        "#66a61e", "#e6ab02", "#a6761d", "#666666",
        "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
        "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6",
        "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
        "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
    ]
    method_keys = sorted(method_keys)
    return {m: palette[i % len(palette)] for i, m in enumerate(method_keys)}


def main():
    parser = argparse.ArgumentParser(description="Interactive plot with tabs per experiment")
    parser.add_argument('--steps_results', type=str,
                        default='results/exp_all/exp_steps_sweep/all_results.json')
    parser.add_argument('--layers_results', type=str,
                        default='results/exp_all/exp_layers_sweep/all_results.json')
    parser.add_argument('--context_results', type=str,
                        default='results/exp_all/exp_context_sweep/all_results.json')
    parser.add_argument('--context_zero_results', type=str,
                        default='results/exp_all/exp_context_sweep_zero/all_results.json')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='interactive_plots.html')
    parser.add_argument('--lowrank_mode', type=str, default='auto',
                        choices=['auto', 'separate'],
                        help='auto = prefer non-block, fallback to block; separate = show both')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_results = _load_if_exists(args.steps_results)
    layers_results = _load_if_exists(args.layers_results)
    context_results = _load_if_exists(args.context_results)
    context_zero_results = _load_if_exists(args.context_zero_results)

    all_methods = set()
    for res in (steps_results, layers_results, context_results, context_zero_results):
        if res is not None:
            all_methods.update(_collect_methods(res, lowrank_mode=args.lowrank_mode))
    all_methods.add("gd")
    color_map = _build_color_map(all_methods)
    color_map["gd"] = "#ED7D31"

    tabs = []
    if steps_results is not None:
        tabs.append(("Steps Sweep", *_build_fig(steps_results, "train_steps", "Training Steps", "Steps Sweep", color_map, args.lowrank_mode)))
    if layers_results is not None:
        tabs.append(("Layers Sweep", *_build_fig(layers_results, "num_layers", "Number of Layers / GD Steps", "Layers Sweep", color_map, args.lowrank_mode)))
    if context_results is not None:
        tabs.append(("Context Sweep", *_build_fig(context_results, "n_points", "In-context Examples (n)", "Context Sweep", color_map, args.lowrank_mode)))
    if context_zero_results is not None:
        tabs.append(("Context Sweep (Zero Train)", *_build_fig(context_zero_results, "n_points", "In-context Examples (n)", "Context Sweep (Zero Train)", color_map, args.lowrank_mode)))

    if not tabs:
        raise SystemExit("No valid results files found; nothing to plot.")

    # De-duplicate tabs by label (keeps first occurrence)
    deduped = []
    seen = set()
    for item in tabs:
        label = item[0]
        if label in seen:
            continue
        seen.add(label)
        deduped.append(item)
    tabs = deduped

    # Build HTML with tabs and embedded Plotly figs
    tab_buttons = []
    tab_contents = []
    fig_payloads = []
    trace_maps = []

    for i, (label, fig, trace_idx, label_map) in enumerate(tabs):
        fig_id = f"fig{i}"
        tab_id = f"tab{i}"
        payload = fig.to_plotly_json()
        fig_payloads.append((fig_id, payload))
        trace_maps.append((fig_id, trace_idx))

        # sidebar checkboxes
        checkbox_lines = []
        for method_key in trace_idx.keys():
            method_label = label_map.get(method_key, method_key)
            color = color_map.get(method_key, "#000000")
            checkbox_lines.append(
                f'<label class="chk">'
                f'<input type="checkbox" checked data-fig="{fig_id}" data-key="{method_key}" '
                f'onchange="toggleMethod(\'{fig_id}\', \'{method_key}\', this.checked)">'
                f'<span class="swatch" style="background:{color}"></span>'
                f'{method_label}</label>'
            )
        sidebar = "\n".join(checkbox_lines)

        tab_buttons.append(
            f'<button class="tab-btn" onclick="openTab(event, \'{tab_id}\')">{label}</button>'
        )
        tab_contents.append(
            f'''
            <div id="{tab_id}" class="tab-content" style="display: {"block" if i==0 else "none"};">
              <div class="tab-grid">
                <div class="sidebar">
                  <div class="sidebar-title">Attention Mechanisms</div>
                  {sidebar}
                  <div class="toggle-all">
                    <button onclick="toggleAll('{fig_id}', true)">Select All</button>
                    <button onclick="toggleAll('{fig_id}', false)">Clear All</button>
                  </div>
                </div>
                <div class="plot-area">
                  <div id="{fig_id}"></div>
                </div>
              </div>
            </div>
            '''
        )

    figs_json = json.dumps({fid: payload for fid, payload in fig_payloads}, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    trace_json = json.dumps({fid: mapping for fid, mapping in trace_maps})

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Interactive Experiment Plots</title>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .tab-btn {{ padding: 8px 12px; margin-right: 6px; border: 1px solid #ccc; background: #f5f5f5; cursor: pointer; }}
    .tab-btn.active {{ background: #e0e0e0; }}
    .tab-content {{ margin-top: 12px; }}
    .tab-grid {{ display: grid; grid-template-columns: 220px 1fr; gap: 16px; align-items: start; }}
    .sidebar {{ border: 1px solid #ddd; padding: 10px; border-radius: 6px; background: #fafafa; }}
    .sidebar-title {{ font-weight: 600; margin-bottom: 8px; }}
    .chk {{ display: block; margin-bottom: 6px; font-size: 14px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 8px 0 6px; border: 1px solid #333; }}
    .toggle-all {{ margin-top: 10px; display: flex; gap: 8px; }}
    .toggle-all button {{ padding: 4px 8px; border: 1px solid #bbb; background: #f3f3f3; cursor: pointer; border-radius: 4px; }}
    .plot-area {{ min-width: 400px; }}
  </style>
  <script>
    let FIGS = {{}};
    let TRACE_MAP = {{}};

    function toggleMethod(figId, methodLabel, isChecked) {{
      const idxs = TRACE_MAP[figId][methodLabel] || [];
      const vis = isChecked ? true : "legendonly";
      Plotly.restyle(figId, {{visible: vis}}, idxs);
    }}

    function openTab(evt, tabId) {{
      var i, tabcontent, tabbuttons;
      tabcontent = document.getElementsByClassName("tab-content");
      for (i = 0; i < tabcontent.length; i++) {{
        tabcontent[i].style.display = "none";
      }}
      tabbuttons = document.getElementsByClassName("tab-btn");
      for (i = 0; i < tabbuttons.length; i++) {{
        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
      }}
      document.getElementById(tabId).style.display = "block";
      evt.currentTarget.className += " active";
    }}
    function toggleAll(figId, checked) {{
      const map = TRACE_MAP[figId] || {{}};
      for (const key of Object.keys(map)) {{
        const boxes = document.querySelectorAll(`input[data-fig='${{figId}}'][data-key='${{key}}']`);
        boxes.forEach(b => b.checked = checked);
        toggleMethod(figId, key, checked);
      }}
    }}
    window.onload = function() {{
      var first = document.getElementsByClassName("tab-btn")[0];
      if (first) {{ first.className += " active"; }}
      for (const [figId, payload] of Object.entries(FIGS)) {{
        Plotly.newPlot(figId, payload.data, payload.layout, {{responsive: true}});
      }}
    }};
  </script>
</head>
<body>
  <h2>Interactive Experiment Plots</h2>
  <div>{"".join(tab_buttons)}</div>
  {"".join(tab_contents)}
  <p>Tip: use the checkboxes to toggle each attention mechanism.</p>
  <script>
    FIGS = {figs_json};
    TRACE_MAP = {trace_json};
  </script>
</body>
</html>
"""

    out_path = output_dir / args.output_name
    out_path.write_text(html)
    print(f"Saved interactive plots to: {out_path}")


if __name__ == "__main__":
    main()
