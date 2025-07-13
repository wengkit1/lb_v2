import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from typing import Dict, List
import re


def extract_hyperparameters(model_name: str) -> Dict:
    """Extract hyperparameters from model names."""
    hps = {}

    # Datamix pattern: n4-g8-en_ratio0.1-cc_ratio0.1-code_ratio0.1
    datamix_patterns = [
        r'en_ratio([\d.]+)',
        r'cc_ratio([\d.]+)',
        r'code_ratio([\d.]+)',
        r'hq_ratio([\d.]+)'
    ]

    for pattern in datamix_patterns:
        match = re.search(pattern, model_name)
        if match:
            param_name = pattern.split('(')[0].replace('_ratio', '_ratio')
            hps[param_name] = float(match.group(1))

    # HPS-sweep pattern: hps-sweep_smc_gemma-3-4b-it_SEQLEN8192_MBS4_N4_FULL_SHARD_LR1e-4_GBS1024_DUR20e9tok_LIGER1_WD1e-5
    hps_patterns = [
        (r'LR([\de.-]+)', 'LR'),
        (r'GBS(\d+)', 'GBS'),
    ]

    for pattern, param_name in hps_patterns:
        match = re.search(pattern, model_name)
        if match:
            try:
                if 'e' in match.group(1) or '.' in match.group(1):
                    hps[param_name] = float(match.group(1))
                else:
                    hps[param_name] = int(match.group(1))
            except ValueError:
                hps[param_name] = match.group(1)

    return hps


def get_all_metrics(exp_data: Dict) -> List[str]:
    """Get all available metrics from lang, competency, and task dataframes."""
    all_metrics = []

    # From lang dataframe
    if 'lang' in exp_data and not exp_data['lang'].empty:
        lang_cols = [col for col in exp_data['lang'].columns if col != 'model']
        all_metrics.extend(lang_cols)

    # From competency dataframe
    if 'competency' in exp_data and not exp_data['competency'].empty:
        comp_cols = [col for col in exp_data['competency'].columns if col != 'model']
        all_metrics.extend(comp_cols)

    # From task dataframe
    if 'task' in exp_data and not exp_data['task'].empty:
        task_cols = [col for col in exp_data['task'].columns if col != 'model']
        all_metrics.extend(task_cols)

    return sorted(list(set(all_metrics)))


def combine_dataframes(exp_data: Dict) -> pd.DataFrame:
    """Combine all dataframes into one with all metrics."""
    combined_df = None

    for df_name in ['lang', 'competency', 'task']:
        if df_name in exp_data and not exp_data[df_name].empty:
            df = exp_data[df_name].reset_index()
            if 'model' not in df.columns:
                df['model'] = df.index

            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='model', how='outer')

    return combined_df


def get_available_hyperparameters(exp_data: Dict) -> List[str]:
    """Extract all unique hyperparameters from the dataframes."""
    combined_df = combine_dataframes(exp_data)
    if combined_df is None or combined_df.empty:
        return []

    # Extract hyperparameters for all models
    combined_df['hyperparams'] = combined_df['model'].apply(extract_hyperparameters)

    # Collect all unique hyperparameter keys
    all_hp_keys = set()
    for hp_dict in combined_df['hyperparams']:
        all_hp_keys.update(hp_dict.keys())

    return sorted(list(all_hp_keys))


def generate_contour_plot(exp_data: Dict, x_param: str, y_param: str, metric: str, x_log: bool = False,
                          y_log: bool = False) -> go.Figure:
    """Generate contour plot with hyperparameters as axes."""
    if not x_param or not y_param or not metric or x_param == y_param:
        return go.Figure()

    combined_df = combine_dataframes(exp_data)
    if combined_df is None or combined_df.empty:
        return go.Figure()

    # Extract hyperparameters
    combined_df['hyperparams'] = combined_df['model'].apply(extract_hyperparameters)

    # Extract the specific hyperparameter values
    combined_df[f'{x_param}_val'] = combined_df['hyperparams'].apply(
        lambda x: x.get(x_param, None)
    )
    combined_df[f'{y_param}_val'] = combined_df['hyperparams'].apply(
        lambda x: x.get(y_param, None)
    )

    # Filter out rows where hyperparameters or metric are missing
    valid_df = combined_df.dropna(subset=[f'{x_param}_val', f'{y_param}_val', metric]).copy()

    if valid_df.empty or len(valid_df) < 3:
        # Not enough points for contour plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Not enough data points for contour plot.<br>Need at least 3 points with valid {x_param}, {y_param}, and {metric} values.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Convert hyperparameter values to numeric
    try:
        x = pd.to_numeric(valid_df[f'{x_param}_val'], errors='coerce')
        y = pd.to_numeric(valid_df[f'{y_param}_val'], errors='coerce')
        z = pd.to_numeric(valid_df[metric], errors='coerce')

        # Remove any NaN values that resulted from conversion
        mask = ~(x.isna() | y.isna() | z.isna())
        x = x[mask]
        y = y[mask]
        z = z[mask]
        valid_df = valid_df[mask]

        if len(x) < 3:
            raise ValueError("Not enough numeric data points")

    except (ValueError, TypeError):
        fig = go.Figure()
        fig.add_annotation(
            text=f"Unable to convert hyperparameters to numeric values.<br>Check that {x_param} and {y_param} contain numbers.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Store original values for hover text
    x_orig, y_orig = x.copy(), y.copy()

    # Apply log scaling if requested
    x_log_scale = x_log and np.all(x > 0)
    y_log_scale = y_log and np.all(y > 0)

    if x_log_scale:
        x = np.log10(x)

    if y_log_scale:
        y = np.log10(y)

    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)

    # Interpolate data onto grid
    try:
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

        # If cubic fails, try linear
        if np.all(np.isnan(zi)):
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

        # If linear also fails, try nearest
        if np.all(np.isnan(zi)):
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='nearest')

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Interpolation failed: {str(e)}<br>Try with different hyperparameters or metric.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

    # Create contour plot
    fig = go.Figure()

    # Add contour
    fig.add_trace(go.Contour(
        x=xi,
        y=yi,
        z=zi,
        colorscale='Viridis',
        contours=dict(
            coloring='fill',
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorbar=dict(title=dict(text=metric, side='right')),
        hovertemplate=f'{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>{metric}: %{{z:.2f}}<extra></extra>'
    ))

    # Add scatter points for actual data
    hover_text = []
    for i, (_, row) in enumerate(valid_df.iterrows()):
        hp_info = []
        for key, value in row['hyperparams'].items():
            hp_info.append(f"{key}: {value}")
        hp_str = "<br>".join(hp_info) if hp_info else "No hyperparams"

        # Use original values in hover text
        hover_text.append(
            f"<b>{row['model']}</b><br>"
            f"{x_param}: {x_orig.iloc[i]}<br>"
            f"{y_param}: {y_orig.iloc[i]}<br>"
            f"{metric}: {row[metric]:.2f}<br>"
            f"<br>{hp_str}"
        )

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=8,
            color='white',
            line=dict(width=2, color='black')
        ),
        name='Data Points',
        hovertemplate='%{text}<extra></extra>',
        text=hover_text
    ))

    # Set up axis labels and tick formatting
    x_title = f"{x_param} (log scale)" if x_log_scale else x_param
    y_title = f"{y_param} (log scale)" if y_log_scale else y_param

    # Custom tick formatting for log scales
    x_tickvals = None
    x_ticktext = None
    y_tickvals = None
    y_ticktext = None

    if x_log_scale:
        # Create nice tick marks for log scale
        x_ticks = np.arange(np.floor(xi.min()), np.ceil(xi.max()) + 1)
        x_tickvals = x_ticks
        x_ticktext = [f"{10 ** tick:.0e}" if tick < -2 or tick > 2 else f"{10 ** tick:.3f}".rstrip('0').rstrip('.') for
                      tick in x_ticks]

    if y_log_scale:
        # Create nice tick marks for log scale
        y_ticks = np.arange(np.floor(yi.min()), np.ceil(yi.max()) + 1)
        y_tickvals = y_ticks
        y_ticktext = [f"{10 ** tick:.0e}" if tick < -2 or tick > 2 else f"{10 ** tick:.3f}".rstrip('0').rstrip('.') for
                      tick in y_ticks]

    # Update layout
    fig.update_layout(
        title=f"Contour Plot: {metric} over {x_param} vs {y_param}",
        xaxis=dict(
            title=x_title,
            tickvals=x_tickvals,
            ticktext=x_ticktext
        ),
        yaxis=dict(
            title=y_title,
            tickvals=y_tickvals,
            ticktext=y_ticktext
        ),
        template="plotly_white",
        height=700,
        showlegend=True
    )

    return fig


def contour_plot_tab(exp_data: Dict, shared_state=None):
    """Create contour plot tab view."""
    with gr.Tab("Contour Plot"):
        if not exp_data or all(df.empty for df in exp_data.values()):
            gr.Markdown("No data available for contour analysis.")
            return

        # Get available hyperparameters and metrics
        available_hyperparams = get_available_hyperparameters(exp_data)
        all_metrics = get_all_metrics(exp_data)

        if len(available_hyperparams) < 2:
            gr.Markdown("At least 2 hyperparameters required for contour analysis.")
            return

        if len(all_metrics) < 1:
            gr.Markdown("At least 1 metric required for contour analysis.")
            return

        with gr.Row():
            with gr.Column():
                x_dropdown = gr.Dropdown(
                    choices=available_hyperparams,
                    label="X-axis Hyperparameter",
                    value=available_hyperparams[0] if available_hyperparams else None,
                    interactive=True
                )
                x_log_checkbox = gr.Checkbox(
                    label="Log scale X-axis",
                    value=False
                )
            with gr.Column():
                y_dropdown = gr.Dropdown(
                    choices=available_hyperparams,
                    label="Y-axis Hyperparameter",
                    value=available_hyperparams[1] if len(available_hyperparams) > 1 else None,
                    interactive=True
                )
                y_log_checkbox = gr.Checkbox(
                    label="Log scale Y-axis",
                    value=False
                )
            with gr.Column():
                metric_dropdown = gr.Dropdown(
                    choices=all_metrics,
                    label="Metric to Contour",
                    value=all_metrics[0] if all_metrics else None,
                    interactive=True
                )

        plot = gr.Plot()

        def update_plot(x_param, y_param, metric, x_log, y_log):
            return generate_contour_plot(exp_data, x_param, y_param, metric, x_log, y_log)

        # Update plot when any dropdown or checkbox changes
        for control in [x_dropdown, y_dropdown, metric_dropdown, x_log_checkbox, y_log_checkbox]:
            control.change(
                fn=update_plot,
                inputs=[x_dropdown, y_dropdown, metric_dropdown, x_log_checkbox, y_log_checkbox],
                outputs=[plot]
            )

        # Initial plot
        if len(available_hyperparams) >= 2 and len(all_metrics) >= 1:
            plot.value = generate_contour_plot(
                exp_data,
                available_hyperparams[0],
                available_hyperparams[1],
                all_metrics[0],
                False,
                False
            )

        gr.Markdown(
            "**Contour Analysis**: Shows how a selected metric varies across different hyperparameter combinations. "
            "The contour lines represent constant values of the metric, while the white dots show actual data points. "
            "Hover over points to see detailed model information and hyperparameter values. "
            "Use the checkboxes to apply log scaling to axes when needed (e.g., for learning rates)."
        )