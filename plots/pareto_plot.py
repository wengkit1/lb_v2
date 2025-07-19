import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from typing import Dict
from .plot_utils import combine_dataframes, extract_hyperparameters, get_all_metrics

def calculate_pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Calculate Pareto front (non-dominated points)."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return df

    # Remove rows with NaN values in either column
    valid_df = df.dropna(subset=[x_col, y_col]).copy()

    if valid_df.empty:
        return valid_df

    # Calculate Pareto front (assuming higher is better for both metrics)
    pareto_points = []

    for i, row in valid_df.iterrows():
        is_dominated = False
        x_val, y_val = row[x_col], row[y_col]

        for j, other_row in valid_df.iterrows():
            if i == j:
                continue
            other_x, other_y = other_row[x_col], other_row[y_col]

            # Point is dominated if another point is better in both dimensions
            if other_x >= x_val and other_y >= y_val and (other_x > x_val or other_y > y_val):
                is_dominated = True
                break

        if not is_dominated:
            pareto_points.append(i)

    return valid_df.loc[pareto_points]


def generate_pareto_plot(exp_data: Dict, x_metric: str, y_metric: str) -> go.Figure:
    """Generate Pareto front plot."""
    if not x_metric or not y_metric or x_metric == y_metric:
        return go.Figure()

    combined_df = combine_dataframes(exp_data)
    if combined_df is None or combined_df.empty:
        return go.Figure()

    # Extract hyperparameters for all models
    combined_df['hyperparams'] = combined_df['model'].apply(extract_hyperparameters)

    # Calculate Pareto front
    pareto_df = calculate_pareto_front(combined_df, x_metric, y_metric)

    if pareto_df.empty:
        return go.Figure()

    # Determine shape/color mapping based on detected hyperparameters
    shape_param = None
    color_param = None

    # Find the most common hyperparameter for shape/color coding
    all_hp_keys = set()
    for hp_dict in pareto_df['hyperparams']:
        all_hp_keys.update(hp_dict.keys())

    if all_hp_keys:
        hp_list = sorted(list(all_hp_keys))
        if len(hp_list) >= 1:
            shape_param = hp_list[0]
        if len(hp_list) >= 2:
            color_param = hp_list[1]

    # Extract values for shape and color mapping
    if shape_param:
        pareto_df[f'{shape_param}_str'] = pareto_df['hyperparams'].apply(
            lambda x: str(x.get(shape_param, 'N/A'))
        )

    if color_param:
        pareto_df[f'{color_param}_str'] = pareto_df['hyperparams'].apply(
            lambda x: str(x.get(color_param, 'N/A'))
        )

    # Create the plot
    fig = go.Figure()

    # Create hover text function
    def create_hover_text(row):
        hp_info = []
        for key, value in row['hyperparams'].items():
            hp_info.append(f"{key}: {value}")
        hp_str = "<br>".join(hp_info) if hp_info else "No hyperparams detected"

        return (
            f"<b>{row['model']}</b><br>"
            f"{x_metric}: {row[x_metric]:.2f}<br>"
            f"{y_metric}: {row[y_metric]:.2f}<br>"
            f"<br>{hp_str}"
        )

    # Plot points with different shapes/colors based on hyperparameters
    if shape_param and color_param:
        # Group by both shape and color parameters
        for (shape_val, color_val), group in pareto_df.groupby([f'{shape_param}_str', f'{color_param}_str']):
            hover_texts = [create_hover_text(row) for _, row in group.iterrows()]
            fig.add_trace(go.Scatter(
                x=group[x_metric],
                y=group[y_metric],
                mode='markers',
                name=f"{shape_param}={shape_val}, {color_param}={color_val}",
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts,
                marker=dict(size=12)
            ))
    elif shape_param:
        # Use only shape parameter
        for shape_val, group in pareto_df.groupby(f'{shape_param}_str'):
            hover_texts = [create_hover_text(row) for _, row in group.iterrows()]
            fig.add_trace(go.Scatter(
                x=group[x_metric],
                y=group[y_metric],
                mode='markers',
                name=f"{shape_param}={shape_val}",
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts,
                marker=dict(size=12)
            ))
    else:
        hover_texts = [create_hover_text(row) for _, row in pareto_df.iterrows()]
        fig.add_trace(go.Scatter(
            x=pareto_df[x_metric],
            y=pareto_df[y_metric],
            mode='markers',
            name='Models',
            hovertemplate='%{text}<extra></extra>',
            text=hover_texts,
            marker=dict(size=12, color='blue')
        ))

    # Add Pareto front line
    if len(pareto_df) > 1:
        # Sort by x-axis for proper line drawing
        sorted_pareto = pareto_df.sort_values(x_metric)
        fig.add_trace(go.Scatter(
            x=sorted_pareto[x_metric],
            y=sorted_pareto[y_metric],
            mode='lines',
            name='Pareto Front',
            line=dict(color='red', width=2, dash='dash'),
            hoverinfo='skip'
        ))

    # Update layout
    fig.update_layout(
        title=f"Pareto Front: {y_metric} vs {x_metric}",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        height=600,
        template="plotly_white",
        showlegend=True
    )

    return fig


def pareto_plot_tab(exp_data: Dict, shared_state=None):
    """Create Pareto front tab view."""
    with gr.Tab("Pareto Plot"):
        if not exp_data or all(df.empty for df in exp_data.values()):
            gr.Markdown("No data available for Pareto front analysis.")
            return

        all_metrics = get_all_metrics(exp_data)

        if len(all_metrics) < 2:
            gr.Markdown("At least 2 metrics required for Pareto front analysis.")
            return

        with gr.Row():
            with gr.Column():
                x_dropdown = gr.Dropdown(
                    choices=all_metrics,
                    label="X-axis Metric",
                    value=all_metrics[0] if all_metrics else None,
                    interactive=True
                )
            with gr.Column():
                y_dropdown = gr.Dropdown(
                    choices=all_metrics,
                    label="Y-axis Metric",
                    value=all_metrics[1] if len(all_metrics) > 1 else None,
                    interactive=True
                )

        plot = gr.Plot()

        def update_plot(x_metric, y_metric):
            return generate_pareto_plot(exp_data, x_metric, y_metric)

        # Update plot when dropdowns change
        x_dropdown.change(
            fn=update_plot,
            inputs=[x_dropdown, y_dropdown],
            outputs=[plot]
        )

        y_dropdown.change(
            fn=update_plot,
            inputs=[x_dropdown, y_dropdown],
            outputs=[plot]
        )

        # Initial plot
        if len(all_metrics) >= 2:
            plot.value = generate_pareto_plot(exp_data, all_metrics[0], all_metrics[1])

        gr.Markdown(
            "**Pareto Front Analysis**: Shows only non-dominated points where no other model "
            "performs better in both selected metrics. The red dashed line connects the Pareto-optimal points. "
            "Different shapes/colors represent different hyperparameter values detected in model names."
        )