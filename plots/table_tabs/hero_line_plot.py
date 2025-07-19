import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import re
from typing import Dict


def extract_hero_run_number(model_name: str) -> int:
    """Extract step number from hero run model names."""
    patterns = [
        r'_ba(\d+)',  # _ba followed by number
        r'step=(\d+)',  # step= followed by number
        r'_(\d+)$',  # _ followed by number at end of string
        r'_(\d+)_',  # _ followed by number followed by _
    ]

    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            return int(match.group(1))
    return 0


def hero_line_plot_tab(df: pd.DataFrame, tab_type: str):
    """Create hero line plot tab function that can be used by TabBuilder."""
    def hero_line_plot_func(*args):
        with gr.Tab("Components Overlay against Baseline"):
            if df.empty:
                gr.Markdown("No data available for training progress plot.")
                return

            df_plot = df.copy()

            df_plot['step_number'] = df_plot['model'].apply(extract_hero_run_number)
            df_plot = df_plot.sort_values('step_number')

            df_plot = df_plot[~df_plot['model'].str.startswith('BASELINE_')]

            if df_plot.empty:
                gr.Markdown("No hero run data available for plotting.")
                return

            fig = go.Figure()

            if tab_type == 'lang':
                language_cols = [col for col in df_plot.columns
                                 if col not in ['model', 'step_number', 'sea_total', 'total']]

                for i, lang_col in enumerate(language_cols):
                    if pd.api.types.is_numeric_dtype(df_plot[lang_col]):
                        fig.add_trace(go.Scatter(
                            x=df_plot['step_number'],
                            y=df_plot[lang_col],
                            mode='lines+markers',
                            name=lang_col,
                            line=dict(width=3),
                            customdata=df_plot['model'],
                            hovertemplate=f'<b>%{{customdata}}</b><br>Step: %{{x}}<br>{lang_col}: %{{y:.4f}}<extra></extra>'
                        ))


                baseline_models = df[
                    df['model'].str.startswith('BASELINE_')] if 'model' in df.columns else pd.DataFrame()


                if not baseline_models.empty:
                    baseline_val = baseline_models['Aggregate'].iloc[0]
                    fig.add_hline(
                        y=baseline_val,
                        line_dash="dash",
                        line_width=2,
                        annotation_text=f"Baseline Aggregate: {baseline_val:.4f}",
                        annotation_position="top right"
                    )

                title = "Language Performance vs Training Steps"

            else:
                available_languages = set()
                non_prefixed_cols = []

                for col in df_plot.columns:
                    if col in ['model', 'step_number', 'sea_total', 'total']:
                        continue
                    elif '_' in col:
                        lang_prefix = col.split('_')[0]
                        available_languages.add(lang_prefix)
                    else:
                        non_prefixed_cols.append(col)

                if non_prefixed_cols:
                    available_languages.add('English')

                available_languages = sorted(list(available_languages))

                if not available_languages:
                    gr.Markdown("No language-specific data available.")
                    return

                language_dropdown = gr.Dropdown(
                    choices=available_languages,
                    value=available_languages[0],
                    label="Select Language",
                    interactive=True
                )

                plot_output = gr.Plot()

                def update_plot(selected_language):
                    fig_inner = go.Figure()

                    lang_cols = [col for col in df_plot.columns
                                 if col.startswith(f'{selected_language}_')]

                    if not lang_cols:
                        lang_cols = non_prefixed_cols

                    baseline_models = df[
                        df['model'].str.startswith('BASELINE_')] if 'model' in df.columns else pd.DataFrame()

                    if not baseline_models.empty:
                        numeric_lang_cols = [col for col in lang_cols
                                             if col in baseline_models.columns and
                                             pd.api.types.is_numeric_dtype(baseline_models[col])]

                        if numeric_lang_cols:
                            baseline_val = baseline_models[numeric_lang_cols].iloc[0].mean()

                        fig_inner.add_hline(
                            y=baseline_val,
                            line_dash="solid",
                            line_color="black",
                            line_width=3,
                            annotation_text=f"Baseline: {baseline_val:.4f}",
                            annotation_position="top right"
                        )

                    for i, component_col in enumerate(lang_cols):
                        if pd.api.types.is_numeric_dtype(df_plot[component_col]):
                            component_name = component_col.replace(f'{selected_language}_', '')

                            fig_inner.add_trace(go.Scatter(
                                x=df_plot['step_number'],
                                y=df_plot[component_col],
                                mode='lines+markers',
                                name=component_name,
                                line=dict(width=2, dash='dash'),
                                marker=dict(size=6),
                                customdata=df_plot['model'],
                                hovertemplate=f'<b>%{{customdata}}</b><br>Step: %{{x}}<br>{component_col}: %{{y:.4f}}<extra></extra>'
                            ))

                    fig_inner.update_layout(
                        title=f"{selected_language.upper()} {tab_type.title()} Components vs Training Steps",
                        xaxis_title="Training Steps",
                        yaxis_title="Score",
                        hovermode='closest',
                        showlegend=True,
                        height=600,
                        template="plotly_white"
                    )

                    return fig_inner

                language_dropdown.change(
                    fn=update_plot,
                    inputs=[language_dropdown],
                    outputs=[plot_output]
                )

                # Initialize plot
                plot_output.value = update_plot(available_languages[0])
                return


            fig.update_layout(
                title=title,
                xaxis_title="Training Steps",
                yaxis_title="Score",
                hovermode='closest',
                showlegend=True,
                height=600,
                template="plotly_white"
            )

            gr.Plot(value=fig)

            gr.Markdown("""
            **Training Progress**: Shows model performance evolution during training.
            - Solid lines: main performance metrics
            - Dashed lines: component metrics (competency/task tabs)
            - Horizontal dashed lines: baseline performance
            """)
    return hero_line_plot_func