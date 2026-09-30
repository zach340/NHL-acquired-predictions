"""
charts.py
=========
Plotly and matplotlib figures (dark theme).
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error

from .config import TARGET_LABELS, TARGETS, get_team_color

BG = "#0e1117"
GRID = "#2d3748"
BAR = "#4a90d9"

FWD_CHART_METRICS = [
    ("pred_game_score_per_game", "Game Score / Game"),
    ("pred_points_per_game",     "Points / Game"),
    ("pred_goals_per_game",      "Goals / Game"),
]


def team_bar_chart(results, actual_team, title, metrics, lower_is_better=(),
                   ascending=True, spacing=0.08, font_size=9):
    """
    One horizontal bar panel per metric ranking every team, with the player's
    actual team highlighted in its colours. The x-axis is zoomed to the data
    range so small team differences are visible; lower-is-better axes are
    reversed. Use the fullscreen icon (⛶) to expand.
    """
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[label for _, label in metrics],
                        horizontal_spacing=spacing)
    primary   = get_team_color(actual_team, "primary")
    secondary = get_team_color(actual_team, "secondary")

    for i, (col, _) in enumerate(metrics, start=1):
        lower_better = col in lower_is_better
        sr   = results.sort_values(col, ascending=(not ascending) if lower_better else ascending)
        is_actual = [t == actual_team for t in sr["player_team"]]
        vals = sr[col].values

        fig.add_trace(go.Bar(
            x=vals, y=sr["player_team"], orientation="h",
            marker_color=[primary if a else BAR for a in is_actual],
            marker_line_color=[secondary if a else BAR for a in is_actual],
            marker_line_width=[2 if a else 0 for a in is_actual],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=i)

        actual_val = float(results.loc[results["player_team"] == actual_team, col].values[0])
        fig.add_vline(x=actual_val, line_color=primary, line_dash="dash", line_width=2, row=1, col=i)

        pad = max((vals.max() - vals.min()) * 0.1, vals.max() * 0.005)
        x_range = [vals.min() - pad, vals.max() + pad]
        fig.update_xaxes(range=x_range[::-1] if lower_better else x_range, row=1, col=i,
                         gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color="#aaa", size=font_size))
        fig.update_yaxes(row=1, col=i, tickfont=dict(color="#aaa", size=font_size), gridcolor=GRID)

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=font_size + 4)),
        paper_bgcolor=BG, plot_bgcolor=BG, height=700,
        margin=dict(l=60, r=20, t=60, b=40), font=dict(color="white"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "white"
        ann.font.size  = font_size + 3
    return fig


def forward_bar_chart(results, actual_team, title):
    return team_bar_chart(results, actual_team, title, FWD_CHART_METRICS)


def importance_chart(models, feature_names, targets=TARGETS, labels=TARGET_LABELS, top_n=15):
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6))
    fig.patch.set_facecolor(BG)
    for ax, target in zip(np.atleast_1d(axes), targets):
        ax.set_facecolor(BG)
        imp = models[target]["global"].feature_importances_
        idx = np.argsort(imp)[-top_n:]
        ax.barh([feature_names[i] for i in idx], imp[idx], color=BAR)
        ax.set_title(labels[target], color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


def scatter(val_df, actual_col, pred_col, label, ax):
    """Predicted vs actual with a y=x reference line and MAE / r annotation."""
    ax.set_facecolor(BG)
    ax.scatter(val_df[pred_col], val_df[actual_col], alpha=0.5, color=BAR, s=20)
    mn = min(val_df[pred_col].min(), val_df[actual_col].min()) - 0.1
    mx = max(val_df[pred_col].max(), val_df[actual_col].max()) + 0.1
    ax.plot([mn, mx], [mn, mx], color="#c8102e", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Predicted {label}", color="white", fontsize=10)
    ax.set_ylabel(f"Actual {label}", color="white", fontsize=10)
    ax.set_title(label, color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    mae  = mean_absolute_error(val_df[actual_col], val_df[pred_col])
    corr = val_df[[actual_col, pred_col]].corr().iloc[0, 1]
    ax.text(0.05, 0.92, f"MAE {mae:.3f}  |  r {corr:.2f}", transform=ax.transAxes, color="white", fontsize=9)


def blank_panel(ax, title, message):
    ax.set_facecolor(BG)
    ax.text(0.5, 0.5, message, ha="center", va="center", color="white", fontsize=12, transform=ax.transAxes)
    ax.set_title(title, color="white")


def calibration_slope(val_df, actual_col, pred_col):
    x, y = val_df[pred_col].values, val_df[actual_col].values
    if len(x) < 2 or np.std(x) == 0:
        return np.nan
    return np.polyfit(x, y, 1)[0]


def elite_segment_stats(val_df, actual_col, pred_col, quantile=0.90):
    """(MAE, bias, n) for players at/above the `quantile` of actual values. Positive bias = overpredicts."""
    if val_df.empty:
        return np.nan, np.nan, 0
    seg = val_df[val_df[actual_col] >= val_df[actual_col].quantile(quantile)]
    if seg.empty:
        return np.nan, np.nan, 0
    return (float(mean_absolute_error(seg[actual_col], seg[pred_col])),
            float((seg[pred_col] - seg[actual_col]).mean()), int(len(seg)))
