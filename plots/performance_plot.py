import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
import gradio as gr
from typing import List
import re
from .utils import flatten, group_by_experiment


def extract_training_duration(model_name):
    """Extract training duration from model name - tries common patterns in order"""
    patterns = [
        r'^(\d+)',  # Number at start
        r'_(\d+)$',  # Number after the last underscore
        r'(\d+)',  # First number found anywhere
    ]

    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            return int(match.group(1))

    return 0

def create_performance_plot(df_grouped: pd.DataFrame, selected_models: List[str] = None) -> Figure:
    """Create a performance plot colored by training duration"""
    if df_grouped.empty:
        fig = px.scatter(title='No data available')
        fig.update_layout(autosize=True, height=800)
        return fig

    # Filter by selected models if any are provided
    if selected_models:
        df_filtered = df_grouped[df_grouped['model'].isin(selected_models)].copy()
    else:
        df_filtered = df_grouped.copy()

    if df_filtered.empty:
        fig = px.scatter(title='No data for selected models')
        fig.update_layout(autosize=True, height=800)
        return fig

    # Extract training duration
    df_filtered['training_duration'] = df_filtered['model'].apply(extract_training_duration)

    # Prepare data for plotting
    df_plot = df_filtered.copy()

    # Handle different column names based on what's available
    if 'sea_total' in df_plot.columns:
        df_plot['SEA Average'] = df_plot['sea_total']
    else:
        df_plot['SEA Average'] = 0

    if 'total' in df_plot.columns:
        df_plot['Overall Average'] = df_plot['total']
    else:
        df_plot['Overall Average'] = 0

    hover_data = ['model', 'training_duration', 'model_family']
    hover_template = ('<b>%{customdata[0]}</b><br>' +
                      'Model Family: %{customdata[2]}<br>' +
                      'SEA Average: %{x:.3f}<br>' +
                      'Overall Average: %{y:.3f}<br>' +
                      'Training Duration: %{customdata[1]}<br>' +
                      '<extra></extra>')

    fig = px.scatter(
        df_plot,
        x='SEA Average',
        y='Overall Average',
        color='training_duration',
        symbol='model_family',
        hover_data=hover_data,
        labels={
            'SEA Average': 'SEA Average Performance',
            'Overall Average': 'Overall Average Performance',
            'training_duration': 'Training Duration',
        },
        title='Model Performance: SEA Average vs Overall Average',
        color_continuous_scale='Burg',
    )

    fig.update_layout(
        xaxis_title="SEA Average Performance",
        yaxis_title="Overall Average Performance",
        hovermode='closest',
        autosize=True,
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0,
            title="Model Family"
        ),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    fig.update_traces(
        marker=dict(size=12),
        hovertemplate=hover_template
    )

    return fig


def performance_plot_tab(unused: dict, shared_state: dict):
    """Performance plot tab that uses shared state"""
    with gr.Tab("Performance Plot"):
        # Get shared components
        model_choice_dropdown = shared_state['model_choice_dropdown']
        grouped_df = shared_state['grouped_df']

        if grouped_df.empty:
            gr.Markdown("No data available for performance plot.")
            return

        gr.Markdown("Performance plot showing selected models (if any) or all models")

        # Create the plot component
        performance_plot = gr.Plot(
            value=create_performance_plot(grouped_df, []),
            label="Performance Plot",
            container=True,
        )

        # Update plot when model selection changes
        def update_performance_plot(selected_models):
            return create_performance_plot(grouped_df, selected_models or [])

        # Connect both change and blur events
        model_choice_dropdown.change(
            fn=update_performance_plot,
            inputs=[model_choice_dropdown],
            outputs=[performance_plot]
        )

        model_choice_dropdown.blur(
            fn=update_performance_plot,
            inputs=[model_choice_dropdown],
            outputs=[performance_plot]
        )