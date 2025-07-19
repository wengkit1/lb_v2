import gradio as gr
from typing import List, Dict
from gradio_leaderboard import Leaderboard, SelectColumns, SearchColumns, ColumnFilter
from pandas import DataFrame

from .hero_line_plot import hero_line_plot_tab
from ..plot_utils import group_columns_by_language
from ...utils import TabBuilder


def create_single_language_tab(lang_name: str, lang_cols: List[str], df_display: DataFrame):
    """Create a single language tab with filtered columns."""

    def language_tab_func(data, *args):
        with gr.Tab(lang_name):
            # Filter to model column + language-specific columns
            subset_cols = ['model'] + lang_cols
            lang_df = df_display[subset_cols].copy()

            clean_cols = {}
            for col in lang_cols:
                clean_name = col.replace(f'{lang_name}_', '', 1)
                clean_cols[col] = clean_name
            lang_df = lang_df.rename(columns=clean_cols)

            select_columns = lang_df.columns.tolist()

            # Create filter columns for numeric data
            filter_columns = []
            for col in lang_df.columns[1:]:  # Skip 'model' column
                if lang_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    filter_columns.append(
                        ColumnFilter(col, type="slider", label=col)
                    )

            Leaderboard(
                value=lang_df,
                select_columns=SelectColumns(
                    default_selection=select_columns,
                    cant_deselect=["model"],
                ),
                search_columns=SearchColumns(
                    primary_column="model",
                    secondary_columns=[],
                    placeholder="Search by model name",
                    label="Search"
                ),
                filter_columns=filter_columns,
            )

    return language_tab_func


def language_breakdown_tab(df_display: DataFrame, lang_groups: Dict, tab_type: str, is_hero_run: bool):
    """Create a language breakdown tab with nested language tabs + hero line plot tab."""

    language_tab_functions = []

    # Add language tabs
    for lang_name, lang_cols in lang_groups.items():
        tab_func = create_single_language_tab(lang_name, lang_cols, df_display)
        language_tab_functions.append(tab_func)

    # Add hero line plot tab (only for hero runs)
    if is_hero_run:
        # Get available metrics for this tab type
        hero_plot_tab = hero_line_plot_tab(df_display, tab_type)
        language_tab_functions.append(hero_plot_tab)

    language_builder = TabBuilder(language_tab_functions)
    language_builder.build()



def overall_tab(exp_data: Dict, *args):
    is_hero_run = exp_data.get('_meta', {}).get('is_hero_run', False)

    with gr.Tab("Overall"):
        df = exp_data.get('lang', None)
        if df is None or df.empty:
            gr.Markdown("No language data available.")
            return

        df_display = df.reset_index()
        select_columns = df_display.columns.tolist()

        # Create filter columns for numeric data
        filter_columns = []
        for col in df_display.columns[1:]:  # Skip first column (likely 'model')
            if df_display[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                filter_columns.append(
                    ColumnFilter(col, type="slider", label=col)
                )

        select_columns_component = SelectColumns(
            default_selection=select_columns,
            cant_deselect=["model"] if "model" in select_columns else []
        )

        search_columns_component = SearchColumns(
            primary_column="model" if "model" in df_display.columns else df_display.columns[0],
            secondary_columns=[],
            placeholder="Search by model name",
            label="Search",
        )

        Leaderboard(
            value=df_display,
            select_columns=select_columns_component,
            search_columns=search_columns_component,
            filter_columns=filter_columns,
        )

        # Add hero line plot for overall/language data (only for hero runs)
        if is_hero_run:
            gr.Markdown("---")
            hero_plot_func = hero_line_plot_tab(df_display, 'lang')
            hero_plot_func(exp_data)


def competency_tab(exp_data: Dict, *args):
    df = exp_data.get('competency')
    with gr.Tab("Competency"):
        if df is None or df.empty:
            gr.Markdown("No competency data available.")
            return

        df_display = df.reset_index()
        lang_groups = group_columns_by_language(df_display)

        if not lang_groups:
            gr.Markdown("No language-specific competency data found.")
            return

        is_hero_run = exp_data.get('_meta', {}).get('is_hero_run', False)
        language_breakdown_tab(df_display, lang_groups, 'competency', is_hero_run)


def task_tab(exp_data: Dict, *args):
    with gr.Tab("Task"):
        df = exp_data.get('task')
        if df is None or df.empty:
            gr.Markdown("No task data available.")
            return

        df_display = df.reset_index()
        lang_groups = group_columns_by_language(df_display)

        if not lang_groups:
            gr.Markdown("No language-specific task data found.")
            return

        is_hero_run = exp_data.get('_meta', {}).get('is_hero_run', False)
        language_breakdown_tab(df_display, lang_groups, 'task', is_hero_run)


def table_view(exp_data: Dict, *args):
    """Main table view tab with overall, competency, and task tabs."""
    table_sub_tabs = [
        overall_tab,
        competency_tab,
        task_tab
    ]
    for tab_func in table_sub_tabs:
        tab_func(exp_data, *args)