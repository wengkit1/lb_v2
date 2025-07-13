import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from ..utils import TabBuilder

def style_delta_df(
        original_df: pd.DataFrame,
        delta_model: str,
        precision=1,
        bold_models=True,
        exclude_columns: list[str] | None = None,
):
    """Style DataFrame for delta comparison.

    Args:
        original_df (pandas.DataFrame): DataFrame to style.
        delta_model (str): Name of model to use for delta comparison.
        precision (int, optional): Number precision. Defaults to 1.
        bold_models (bool, optional): Whether to bold the delta model. Defaults to True.
        exclude_columns (list[str] | None, optional): Columns to exclude from delta
            highlighting. Defaults to None.

    Returns:
        Styled dataframe
    """
    if exclude_columns is None:
        exclude_columns = []

    delta_df = original_df.copy()

    # Get numeric columns excluding specified ones
    delta_columns = [
        col
        for col in delta_df.select_dtypes(include="number").columns
        if col not in exclude_columns
    ]

    # Calculate deltas
    for col in delta_columns:
        if delta_model not in delta_df.index:
            delta_df[col] = pd.Series([pd.NA] * len(delta_df))
        else:
            baseline_value = delta_df.loc[delta_model, col]
            delta_df[col] = delta_df[col] - baseline_value

    def styling_fn(styler):
        cmap = plt.get_cmap("PiYG")
        cmap.set_bad("white", 1.0)

        for col in delta_columns:
            if delta_df[col].notna().any():
                max_value = max(abs(delta_df[col].max()), abs(delta_df[col].min()))
                if max_value > 0:
                    styler.background_gradient(
                        cmap=cmap,
                        vmin=-max_value,
                        vmax=max_value,
                        subset=pd.IndexSlice[
                            delta_df.index[delta_df.index != delta_model], col
                        ],
                    )

        if bold_models:
            bolded_models = [delta_model]
            styler.set_properties(
                **{"font-weight": "bold"},
                subset=pd.IndexSlice[
                       delta_df.index[delta_df.index.isin(bolded_models)], :
                       ],
            )

        styler.set_properties(
            **{"background-color": "white", "color": "black"},
            subset=pd.IndexSlice[
                delta_df.index[delta_df.index == delta_model],
                delta_columns,
            ],
        )

        # Format the numbers
        styler.format(
            lambda x: "{:+,.1f}".format(x) if pd.notna(x) else "-",
            na_rep="-",
            subset=pd.IndexSlice[
                delta_df.index[delta_df.index != delta_model], delta_columns
            ],
        )
        styler.format(
            na_rep="-",
            precision=precision,
            subset=pd.IndexSlice[
                delta_df.index[delta_df.index == delta_model], delta_columns
            ],
        )

        return styler

    style = delta_df.style.pipe(styling_fn)

    # Restore original values for the baseline model
    if delta_model in original_df.index:
        def insert_original_values(row):
            if row.name == delta_model:
                return original_df.loc[delta_model]
            else:
                return row

        style.data = style.data.apply(insert_original_values, axis=1)

    return style


def get_baseline_models(df: pd.DataFrame) -> list[str]:
    """Extract baseline model names from DataFrame index.

    Args:
        df: DataFrame with model names as index

    Returns:
        List of baseline model names
    """
    return [model for model in df.index if model.startswith('BASELINE_')]


def create_delta_comparison_plot(exp_data: dict, df_type: str = "lang"):
    """Create delta comparison plot for experiment data.

    Args:
        exp_data: Experiment data dictionary containing 'lang', 'competency', 'task' DataFrames
        df_type: Which DataFrame to use ('lang', 'competency', or 'task')
    """
    # Get the specified DataFrame
    df = exp_data.get(df_type, pd.DataFrame())

    if df.empty:
        gr.Markdown("No data available for delta comparison")
        return

    # Reset index to make model names a column for compatibility
    df_reset = df.reset_index()

    # Ensure we have a 'model' column
    if 'model' not in df_reset.columns and df.index.name:
        df_reset = df_reset.rename(columns={df.index.name: 'model'})
    elif 'model' not in df_reset.columns:
        df_reset.insert(0, 'model', df.index)

    gr.Markdown("Select model for delta comparison")

    model_choices = df_reset["model"].tolist()
    baseline_models = get_baseline_models(df)

    # Default to first baseline if available, otherwise first model
    default_model = baseline_models[0] if baseline_models else (model_choices[0] if model_choices else "")

    delta_dropdown = gr.Dropdown(
        choices=model_choices,
        value=default_model,
        label="Select model for delta comparison",
        interactive=True,
        container=False,
    )

    # Set column widths based on number of columns
    widths = [330] + [110] * (len(df_reset.columns) - 1)
    column_widths = widths[:len(df_reset.columns)]

    # Initial styled dataframe (convert back to indexed format for styling)
    df_for_styling = df_reset.set_index('model')
    initial_styled_df = style_delta_df(df_for_styling, default_model)

    df_display = gr.DataFrame(
        initial_styled_df,
        max_height=800,
        column_widths=column_widths,
    )

    def update_delta_table(selected_model):
        if not selected_model:
            return df_reset

        # Convert to indexed format for styling
        df_for_styling = df_reset.set_index('model')
        styled_df = style_delta_df(df_for_styling, selected_model)
        return styled_df

    delta_dropdown.input(
        fn=update_delta_table,
        inputs=[delta_dropdown],
        outputs=[df_display],
    )

    gr.Markdown(
        """
        **Delta Comparison**: Shows the difference between each model and the selected baseline model.
        - Green cells indicate better performance than baseline
        - Red cells indicate worse performance than baseline  
        - The baseline model row shows original values (not deltas)
        """
    )


def language_performance_tab(exp_data: dict):
    """Language-level comparison tab"""
    with gr.Tab("Language Performance"):
        create_delta_comparison_plot(exp_data, "lang")


def competency_performance_tab(exp_data: dict):
    """Competency-level comparison tab"""
    with gr.Tab("Competency Performance"):
        create_delta_comparison_plot(exp_data, "competency")


def task_performance_tab(exp_data: dict):
    """Task-level comparison tab"""
    with gr.Tab("Task Performance"):
        create_delta_comparison_plot(exp_data, "task")


def delta_comparison_tab(exp_data: dict):
    """Create a complete delta comparison tab with sub-tabs for different data types."""
    with gr.Tab("Delta Comparison"):
        # Define the sub-tab structure
        delta_sub_tabs = [
            language_performance_tab,
            competency_performance_tab,
            task_performance_tab
        ]

        # Build the sub-tabs using TabBuilder
        sub_builder = TabBuilder(delta_sub_tabs)
        sub_builder.build(exp_data)