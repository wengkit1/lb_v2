from typing import Dict

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from ..utils import TabBuilder
from .plot_utils import model_selector, model_selector_for_experiment


def style_delta_df(
        original_df: pd.DataFrame,
        delta_model: str,
        precision=1,
        bold_models=True,
        exclude_columns: list[str] | None = None,
):
    """Style DataFrame for delta comparison.

    Args:
        original_df (pandas.DataFrame): DataFrame to style with model names as index.
        delta_model (str): Name of the model used for delta comparison.
        precision (int, optional): Number precision. defaults to 1.
        bold_models (bool, optional): Whether to bold the delta model. defaults to True.
        exclude_columns (list[str] | None, optional): Columns to exclude from delta
            highlighting. defaults to None.

    Returns:
        Styled dataframe
    """
    if exclude_columns is None:
        exclude_columns = []

    delta_df = original_df.reset_index()

    delta_columns = [
        col
        for col in delta_df.select_dtypes(include="number").columns
        if col not in exclude_columns and col != 'model'
    ]

    # Calculate deltas
    for col in delta_columns:
        if delta_model not in delta_df['model'].values:
            delta_df[col] = pd.Series([pd.NA] * len(delta_df))
        else:
            baseline_value = delta_df[delta_df['model'] == delta_model][col].iloc[0]
            delta_df[col] = delta_df[col] - baseline_value

    def styling_fn(styler):
        cmap = plt.get_cmap('PiYG')
        cmap.set_bad("white", 1.0)

        for col in delta_columns:
            if delta_df[col].notna().any():
                max_value = max(abs(delta_df[col].max()), abs(delta_df[col].min()))
                if max_value > 0:
                    non_baseline_mask = delta_df['model'] != delta_model
                    if non_baseline_mask.any():
                        styler.background_gradient(
                            cmap=cmap,
                            vmin=-max_value,
                            vmax=max_value,
                            subset=pd.IndexSlice[
                                delta_df.index[non_baseline_mask], col
                            ],
                        )

        if bold_models:
            # Bold the baseline model row
            baseline_mask = delta_df['model'] == delta_model
            if baseline_mask.any():
                styler.set_properties(
                    **{"font-weight": "bold"},
                    subset=pd.IndexSlice[
                           delta_df.index[baseline_mask], :
                           ],
                )

        # Set baseline model background to white
        baseline_mask = delta_df['model'] == delta_model
        if baseline_mask.any():
            styler.set_properties(
                **{"background-color": "white", "color": "black"},
                subset=pd.IndexSlice[
                    delta_df.index[baseline_mask],
                    delta_columns,
                ],
            )

        # Format the numbers
        non_baseline_mask = delta_df['model'] != delta_model
        if non_baseline_mask.any():
            styler.format(
                lambda x: "{:+,.1f}".format(x) if pd.notna(x) else "-",
                na_rep="-",
                subset=pd.IndexSlice[
                    delta_df.index[non_baseline_mask], delta_columns
                ],
            )

        baseline_mask = delta_df['model'] == delta_model
        if baseline_mask.any():
            styler.format(
                na_rep="-",
                precision=precision,
                subset=pd.IndexSlice[
                    delta_df.index[baseline_mask], delta_columns
                ],
            )

        return styler

    style = delta_df.style.pipe(styling_fn)

    # Restore original values for the baseline model
    if delta_model in original_df.index:
        baseline_mask = delta_df['model'] == delta_model
        if baseline_mask.any():
            baseline_idx = delta_df.index[baseline_mask][0]
            for col in delta_columns:
                if col in original_df.columns:
                    original_value = original_df.loc[delta_model, col]
                    style.data.iloc[baseline_idx, style.data.columns.get_loc(col)] = original_value

    return style


def get_baseline_models(df: pd.DataFrame) -> list[str]:
    """Extract baseline model names from the DataFrame index.

    Args:
        df: DataFrame with model names as index

    Returns:
        List of baseline model names
    """
    return [model for model in df.index if model.startswith('BASELINE_')]


def create_delta_comparison_plot(df: pd.DataFrame, shared_state: Dict):
    """Create a delta comparison plot for experiment data.

    Args:
        df: DataFrame to create delta comparison for
        shared_state: Shared state containing model selector components
    """
    if df.empty:
        gr.Markdown("No data available for delta comparison")
        return

    # Get the shared model dropdown from shared_state
    delta_dropdown = shared_state.get('model_choice_dropdown')

    if not delta_dropdown:
        gr.Markdown("Model selector not available")
        return

    # DON'T create a new dropdown - use the existing one from shared_state
    # The dropdown should already be rendered in the parent tab context

    # Get models for this specific DataFrame
    model_choices = df.index.tolist()
    baseline_models = get_baseline_models(df)
    default_model = baseline_models[0] if baseline_models else (model_choices[0] if model_choices else "")

    # Set column widths based on the number of columns
    widths = [330] + [110] * len(df.columns)
    column_widths = widths

    # Initial styled dataframe using the default model
    initial_styled_df = style_delta_df(df, default_model)

    df_display = gr.DataFrame(
        initial_styled_df,
        max_height=800,
        column_widths=column_widths,
    )

    def update_delta_table(selected_model):
        if not selected_model:
            return df
        styled_df = style_delta_df(df, selected_model)
        return styled_df

    # Connect the existing shared dropdown to this dataframe
    delta_dropdown.change(
        fn=update_delta_table,
        inputs=[delta_dropdown],
        outputs=[df_display],
    )

    gr.Markdown(
        """
        **Delta Comparison**: Shows the difference between each model and the selected baseline model.
        - Greener cells indicate better performance than baseline
        - Redder cells indicate worse performance than baseline  
        - The baseline model row shows original values (not deltas)
        """
    )


def language_performance_tab(exp_data: Dict, shared_state):
    """Language-level comparison tab"""
    with gr.Tab("Language"):
        df_lang = exp_data.get("lang", pd.DataFrame())
        create_delta_comparison_plot(df_lang, shared_state=shared_state)


def competency_performance_tab(exp_data: Dict, shared_state):
    """Competency-level comparison tab"""
    with gr.Tab("Competency"):
        df_competency = exp_data.get("competency", pd.DataFrame())
        create_delta_comparison_plot(df_competency, shared_state=shared_state)


def task_performance_tab(exp_data: Dict, shared_state):
    """Task-level comparison tab"""
    with gr.Tab("Task"):
        df_task = exp_data.get("task", pd.DataFrame())
        create_delta_comparison_plot(df_task, shared_state=shared_state)

def delta_comparison_plot_tab(exp_data: Dict, shared_state: Dict):
    """Delta comparison tab"""
    with gr.Tab("Delta Comparison"):
        shared_state = {}
        model_selector_for_experiment(exp_data, shared_state)

        gr.Markdown("Select model for delta comparison")
        shared_state.get('model_choice_dropdown')

        delta_sub_tabs = [
            language_performance_tab,
            competency_performance_tab,
            task_performance_tab
        ]

        delta_builder = TabBuilder(tabs=delta_sub_tabs, shared_state=shared_state)
        delta_builder.build(exp_data)