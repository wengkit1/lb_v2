import gradio as gr
from pandas import DataFrame
from gradio_leaderboard import Leaderboard, SelectColumns, SearchColumns, ColumnFilter
from ..utils import TabBuilder


def group_columns_by_language(df: DataFrame):
    """Group columns by language prefix."""
    lang_groups = {}
    # Since names are already cleaned, we look for the actual language names
    lang_order = ['English', 'Indonesian', 'Vietnamese', 'Thai', 'Tamil', 'Tagalog']

    # Initialize with empty lists
    for lang in lang_order:
        lang_groups[lang] = []

    for col in df.columns:
        if col == 'model':
            continue

        # Check if column starts with language name
        lang_prefix = None
        for lang_name in lang_order:
            if col.startswith(f'{lang_name}_'):
                lang_prefix = lang_name
                break

        if lang_prefix:
            lang_groups[lang_prefix].append(col)
        else:
            # Default to 'English' for columns without language prefix
            lang_groups['English'].append(col)

    # Return only languages that have columns
    return {lang: cols for lang, cols in lang_groups.items() if cols}


def create_single_language_tab(lang_name, lang_cols, df_display: DataFrame):
    """Create a single language tab with filtered columns."""

    def language_tab_func(data, *args):  # Use *args for flexible parameters
        with gr.Tab(lang_name):
            # Filter to model column + language-specific columns
            subset_cols = ['model'] + lang_cols
            lang_df = df_display[subset_cols].copy()

            # Clean column names by removing language prefix
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


def create_language_breakdown_tab(df_display: DataFrame, lang_groups):
    """Create a language breakdown tab with nested language tabs."""

    def language_breakdown_func(data, *args):  # Use *args for flexible parameters
        language_tab_functions = []
        for lang_name, lang_cols in lang_groups.items():
            tab_func = create_single_language_tab(lang_name, lang_cols, df_display)
            language_tab_functions.append(tab_func)

        # Use TabBuilder for nested language tabs
        language_builder = TabBuilder(language_tab_functions)
        language_builder.build(data)

    return language_breakdown_func


def overall_tab(exp_data: dict, *args):  # Use *args for flexible parameters
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


def competency_tab(exp_data: dict, *args):
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

        # Create language breakdown with nested tabs
        breakdown_tab = create_language_breakdown_tab(df_display, lang_groups)
        breakdown_tab(exp_data, *args)  # Pass along any additional args


def task_tab(exp_data: dict, *args):
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

        # Create language breakdown with nested tabs
        breakdown_tab = create_language_breakdown_tab(df_display, lang_groups)
        breakdown_tab(exp_data, *args)


def table_view(exp_data, *args):
    """Main table view tab with overall, competency, and task breakdowns."""
    table_sub_tabs = [
        overall_tab,
        competency_tab,
        task_tab
    ]
    for tab_func in table_sub_tabs:
        tab_func(exp_data, *args)