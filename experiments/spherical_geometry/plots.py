"""Plotly figures with explicit units; static export failures are recorded."""
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def style(fig, title, x=None, y=None):
    fig.update_layout(template="plotly_white", title=title, font=dict(family="Arial", size=16),
                      width=1000, height=580, margin=dict(l=85, r=45, t=85, b=75))
    if x:
        fig.update_xaxes(title=x)
    if y:
        fig.update_yaxes(title=y)
    return fig


def pose_figures(result, pairs, coverage):
    xyz = result["xyz"]
    fig = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                               mode="lines+markers", marker=dict(size=3),
                               text=result["source_ids"], name="Camera centres"))
    style(fig, "Camera trajectory — recorded poses")
    fig.update_layout(scene=dict(xaxis_title="World X (m)", yaxis_title="World Y (m)",
                                 zaxis_title="World Z (m)", aspectmode="data"))
    figures = {"trajectory": fig}
    for key, label in [("step_distance_m", "Translation (m)"),
                       ("rotation_step_deg", "SO(3) rotation (degrees)")]:
        fig = go.Figure(go.Scatter(x=result["source_ids"][1:], y=result[key], mode="lines+markers"))
        figures[key] = style(fig, "Adjacent retained-frame motion", "Destination source frame ID", label)
    fig = go.Figure()
    for label, key in [("Eligible", "eligible_pairs"), ("Selected", "selected_pairs")]:
        fig.add_bar(x=[str(r["bin"]) for r in coverage], y=[r[key] for r in coverage], name=label)
    figures["baseline_coverage"] = style(fig, "Metric baseline bin coverage", "Baseline bin", "Pair count")
    d = np.sort(result["step_distance_m"])
    fig = go.Figure(go.Scatter(x=d, y=np.arange(1, len(d)+1)/max(1, len(d)), mode="lines", name="Adjacent retained"))
    if len(pairs):
        d = np.sort(pairs.baseline_m)
        fig.add_scatter(x=d, y=np.arange(1, len(d)+1)/len(d), mode="lines", name="Selected pairs (stratified)")
    figures["baseline_cdf"] = style(fig, "Baseline empirical distributions", "Baseline (m)", "Cumulative fraction")
    return figures


def erp_map(values, longitude, latitude, title, unit):
    fig = go.Figure(go.Heatmap(z=values, x=longitude, y=latitude,
                               colorbar=dict(title=unit), colorscale="Viridis"))
    return style(fig, title, "Longitude (degrees)", "Latitude (degrees)")


def motion_figures(matches, summaries):
    figures = {}
    if not matches.empty:
        fig = px.scatter(matches, x="longitude_deg", y="latitude_deg",
                         color="residual_deg", facet_col="pair_id", facet_col_wrap=3,
                         hover_data=["raw_motion_deg", "frame_i", "frame_j"],
                         color_continuous_scale="Viridis",
                         labels={"longitude_deg": "Longitude (degrees)",
                                 "latitude_deg": "Latitude (degrees)",
                                 "residual_deg": "Residual (deg)"})
        # Sparse samples only: never interpolate unobserved pixels into a dense map.
        style(fig, "Sparse rotation-compensated residuals — tentative feature matches")
        fig.update_layout(height=max(580, 300*int(np.ceil(matches.pair_id.nunique()/3))))
        figures["sparse_parallax_map"] = fig
        grouped = matches.groupby("latitude_bin", observed=True).agg(
            latitude_deg=("latitude_deg", "mean"), residual_deg=("residual_deg", "median"),
            count=("residual_deg", "size")).reset_index()
        fig = px.line(grouped, x="latitude_deg", y="residual_deg", markers=True, hover_data=["count"])
        figures["latitude_residual_profile"] = style(
            fig, "Feature-sampled residual profile (not area-weighted error)",
            "Mean sampled latitude (degrees)", "Median residual (degrees)")
    good = summaries[summaries.status == "measured"] if not summaries.empty else summaries
    if not good.empty:
        fig = px.scatter(good, x="baseline_m", y="median_residual_deg", color="rotation_deg",
                         hover_data=["pair_id", "matches", "longitude_sectors"])
        figures["baseline_parallax"] = style(fig, "Residual versus metric baseline",
                                            "Baseline (m)", "Median residual (degrees)")
    return figures


def save_figures(figures, output_dir, static_formats=("png", "pdf")):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = []
    for name, fig in figures.items():
        fig.write_html(out / (name + ".html"), include_plotlyjs=True)
        for fmt in static_formats:
            if fmt not in ("png", "pdf"):
                raise ValueError("Static format must be png or pdf")
            try:
                fig.write_image(out / (name + "." + fmt), scale=3)
                status.append(dict(figure=name, format=fmt, status="saved"))
            except Exception as exc:
                status.append(dict(figure=name, format=fmt, status="failed", reason=str(exc)))
    return status
