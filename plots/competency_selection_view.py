import pandas as pd
import gradio as gr
from gradio_leaderboard import Leaderboard, SearchColumns, ColumnFilter
from typing import Dict
from .plot_utils import group_columns_by_language

def re_aggregate_scores(competency_results: pd.DataFrame, selected_competencies_by_lang: Dict):
    """Re-aggregate language scores by averaging selected competencies"""
    # Start with the original language scores structure
    updated_scores = pd.DataFrame(index=competency_results.index)
    updated_scores['model'] = competency_results.index

    # Track which languages were updated for aggregate calculation
    updated_languages = []

    for lang, selected_competencies in selected_competencies_by_lang.items():
        if not selected_competencies:
            updated_scores[lang.lower()] = 0.0
            continue

        # Get competency columns for this language
        selected_columns = [col for col in competency_results.columns
                            if col in selected_competencies]

        if selected_columns:
            # Calculate mean of selected competencies
            lang_scores = competency_results[selected_columns].fillna(0.0).mean(axis=1).round(2)
            updated_scores[lang.lower()] = lang_scores
            updated_languages.append(lang.lower())

    sea_languages = [lang for lang in updated_languages if lang != 'english']
    if sea_languages:
        updated_scores['sea_total'] = updated_scores[sea_languages].mean(axis=1).round(2)
    else:
        updated_scores['sea_total'] = 0.0

    if 'sea_total' in updated_scores.columns and 'english' in updated_scores.columns:
        updated_scores['total'] = ((updated_scores['sea_total'] + updated_scores['english']) / 2).round(2)
    elif 'sea_total' in updated_scores.columns:
        updated_scores['total'] = updated_scores['sea_total']
    elif 'english' in updated_scores.columns:
        updated_scores['total'] = updated_scores['english']
    else:
        updated_scores['total'] = 0.0

    return updated_scores


def competency_selection_tab(exp_data: Dict, shared_state: Dict = None):
    """Create the competency selection tab with include/exclude functionality"""

    with gr.Tab("Incl/Excl Competencies"):
        competency_df = exp_data.get('competency', pd.DataFrame())

        if competency_df.empty:
            gr.Markdown("No competency data available for selection.")
            return

        competency_reset = competency_df.reset_index()

        lang_grouped = group_columns_by_language(competency_reset)

        if not lang_grouped:
            gr.Markdown("No language-specific competency data found.")
            return

        competency_selectors = {}

        with gr.Column():
            for lang, competencies in lang_grouped.items():
                with gr.Group():
                    gr.Markdown(f"**{lang}**")
                    competency_selectors[lang] = gr.CheckboxGroup(
                        choices=competencies,
                        value=competencies,
                        label=f"Competencies for {lang}",
                        interactive=True
                    )

        # Initial aggregated scores
        initial_scores = re_aggregate_scores(competency_reset.set_index('model'),
                                             {lang: comps for lang, comps in lang_grouped.items()})

        # Create filter columns for numeric data
        filter_columns = []
        for col in initial_scores.columns[1:]:  # Skip 'model' column
            if initial_scores[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                filter_columns.append(
                    ColumnFilter(col, type="slider", label=col)
                )

        # Create leaderboard component
        leaderboard_component = Leaderboard(
            value=initial_scores,
            search_columns=SearchColumns(
                primary_column="model" if "model" in initial_scores.columns else initial_scores.columns[0],
                secondary_columns=[],
                placeholder="Search by model name",
                label="Search"
            ),
            filter_columns=filter_columns,
        )

        def update_leaderboard(*selected_competencies_args):
            """Update leaderboard based on selected competencies"""
            # Map the arguments back to language names
            selected_by_lang = {
                lang: selected_competencies_args[i]
                for i, lang in enumerate(lang_grouped.keys())
            }

            # Re-aggregate the scores
            updated_scores = re_aggregate_scores(competency_reset.set_index('model'), selected_by_lang)

            # Ensure all values are filled
            updated_scores = updated_scores.fillna(0.0)

            if updated_scores.empty:
                # Return empty dataframe with proper structure
                empty_df = pd.DataFrame({'model': ['No data'], 'score': [0.0]})
                return empty_df

            return updated_scores

        # Connect all selectors to update function
        all_selectors = list(competency_selectors.values())
        for selector in all_selectors:
            selector.change(
                fn=update_leaderboard,
                inputs=all_selectors,
                outputs=[leaderboard_component]
            )

        gr.Markdown("""
        **Include/Exclude Competencies**: Select which competencies to include in the language scores calculation.
        - Uncheck competencies to exclude them from the average
        - Language scores are recalculated as the average of selected competencies
        - SEA Total and Overall Total are automatically updated based on your selections
        """)