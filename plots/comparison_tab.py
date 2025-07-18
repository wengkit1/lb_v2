import gradio as gr
from gradio_leaderboard import Leaderboard, SelectColumns, SearchColumns
from pandas.core.interchange.dataframe_protocol import DataFrame

from .plot_utils import model_selector
from ..utils import TabBuilder
from typing import Dict
from .performance_plot import performance_plot_tab


def comparison_table_plot(flattened_df: DataFrame, shared_state: Dict = None):
    def table_plot_func(unused):
        model_choice_dropdown = shared_state['model_choice_dropdown']
        selected_models_state = shared_state['selected_models_state']

        with gr.Tab("Select & Compare"):
            gr.Markdown("Select models for comparison (click out of the dropdown to apply)")

            @gr.render(
                inputs=[model_choice_dropdown, selected_models_state],
                triggers=[model_choice_dropdown.blur],
            )

            def filter_comparison_models(selected_models):
                if not selected_models:
                    return

                # Filter the DataFrame
                mask = flattened_df['model'].isin(selected_models)
                filtered_df = flattened_df[mask].copy()

                # Rename columns for better display
                display_columns = {}
                if 'sea_total' in filtered_df.columns:
                    display_columns['sea_total'] = 'SEA Average'
                if 'total' in filtered_df.columns:
                    display_columns['total'] = 'Overall Average'

                df_display = filtered_df.rename(columns=display_columns)
                select_columns = df_display.columns.tolist()

                Leaderboard(
                    value=df_display,
                    select_columns=SelectColumns(
                        default_selection=select_columns,
                        cant_deselect=["model"]
                    ),
                    search_columns=SearchColumns(
                        primary_column="model",
                        secondary_columns=[],
                        placeholder="Search by model name",
                        label="Search"
                    )
                )

    return table_plot_func


def comparison_table_tab(unused, shared_state):
    """Table comparison tab that uses shared state"""
    with gr.Tab("Select & Compare"):
        # Get shared components
        model_choice_dropdown = shared_state['model_choice_dropdown']
        flattened_df = shared_state['flattened_df']

        if flattened_df.empty:
            gr.Markdown("No data available for comparison.")
            return

        @gr.render(
            inputs=model_choice_dropdown,
            triggers=[model_choice_dropdown.blur, model_choice_dropdown.change]
        )
        def filter_comparison_models(selected_models):
            if not selected_models:
                gr.Markdown("Please select models from the dropdown above.")
                return

            # Filter the DataFrame
            mask = flattened_df['model'].isin(selected_models)
            filtered_df = flattened_df[mask].copy()

            if filtered_df.empty:
                gr.Markdown("No data found for selected models.")
                return

            # Rename columns for better display
            display_columns = {}
            if 'sea_total' in filtered_df.columns:
                display_columns['sea_total'] = 'SEA Average'
            if 'total' in filtered_df.columns:
                display_columns['total'] = 'Overall Average'

            df_display = filtered_df.rename(columns=display_columns)
            select_columns = df_display.columns.tolist()

            Leaderboard(
                value=df_display,
                select_columns=SelectColumns(
                    default_selection=select_columns,
                    cant_deselect=["model"]
                ),
                search_columns=SearchColumns(
                    primary_column="model",
                    secondary_columns=[],
                    placeholder="Search by model name",
                    label="Search"
                )
            )


def comparison_tab(results: Dict):
    """Main comparison tab function that works with TabBuilder"""

    def comparison_tab_func(*args):
        with gr.Tab("Comparison"):
            shared_state = {}
            model_selector(results, shared_state)
            tabs = [
                comparison_table_tab,
                performance_plot_tab
            ]

            builder = TabBuilder(tabs=tabs, shared_state=shared_state)
            builder.build()

    return comparison_tab_func
