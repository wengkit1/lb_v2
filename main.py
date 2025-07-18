from pandas import DataFrame

from leaderboard_v2.constants.text_blobs import ABOUT_SEAHELM, ADDITIONAL_INFORMATION, ABOUT_AISG, SCORE_CALCULATION
from leaderboard_v2.plots.comparison_tab import comparison_tab
from leaderboard_v2.plots.contour_plot import contour_plot_tab

from .plots.pareto_plot import pareto_plot_tab
from .plots.table_view import table_view
from .plots.delta_comparison import delta_comparison_plot_tab
from .plots.competency_selection_view import competency_selection_tab

from leaderboard_v2.utils import process_runs, load_config, TabBuilder

from .constants.constants import LANGUAGE_NAMES
import gradio as gr

slider_css = """
        .gradio-container .form {
            overflow-y: auto !important;
            max-height: 800px !important;
        }
        """
dynamic_average_js = f"function() {{ {open('leaderboard_v2/js/dynamic_averages.js').read()} }}"

def clean_column_names(df_dict: dict[str, DataFrame]):
    """Apply language name mapping to dataframe columns for all parts of df_dict"""

    cleaned_dict = {}

    for key, df in df_dict.items():
        if df.empty:
            cleaned_dict[key] = df
            continue

        df_copy = df.copy()
        columns = {k.lower(): v for k, v in LANGUAGE_NAMES.items()}
        if key == 'lang':
            df_copy = df_copy.rename(columns=columns)
        elif key in ['competency', 'task']:
            column_mapping = {}
            for col in df_copy.columns:
                if '_' in col:
                    lang_code, rest = col.split('_', 1)
                    if lang_code in columns:
                        new_col = f"{columns[lang_code]}_{rest}"
                        column_mapping[col] = new_col
                    else:
                        column_mapping[col] = col
                else:
                    column_mapping[col] = col

            df_copy = df_copy.rename(columns=column_mapping)

        cleaned_dict[key] = df_copy

    return cleaned_dict


def get_experiments_only(results):
    """Filter to get only experiments (no baselines)"""
    return {k: v for k, v in results.items()
            if not k.startswith('BASELINE_') and not v.get('_meta', {}).get('is_baseline', False)}


def create_main_tabs(results_dict: dict[str, dict[str, DataFrame]]):
    experiment_tab_structure = [
        delta_comparison_plot_tab,
        table_view,
        pareto_plot_tab,
        contour_plot_tab,
        competency_selection_tab
    ]
    experiments = get_experiments_only(results_dict)
    experiment_tabs = []

    for exp_name, exp_data in experiments.items():
        filtered_data = {k: v for k, v in exp_data.items() if k != '_meta'}
        cleaned_data = clean_column_names(filtered_data)
        experiment_tab = TabBuilder(
            data=cleaned_data,
            tabs=experiment_tab_structure,
            tab_name=exp_name
        )
        experiment_tabs.append(experiment_tab)

    all_tabs = experiment_tabs + [comparison_tab(results_dict)]
    main_builder = TabBuilder(all_tabs)
    main_builder.build()

def create_gradio_app(results_dict):
    """Main entry point - create the Gradio app"""
    with gr.Blocks(css=slider_css, js=dynamic_average_js) as demo:
        gr.Markdown("<br>")
        gr.Markdown(
            "<h1 style='margin: 0; padding: 0; font-size: 2.5em;'>🌏 SEA HELM (Internal)</h1>"
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(ABOUT_SEAHELM)
            with gr.Column(scale=1):
                gr.Markdown(
                    "<div style='text-align: right;'>Hosted by the AI Products Team @ AI Singapore</div>"
                )
        gr.Markdown("<br>")

        create_main_tabs(results_dict)

        gr.Markdown("<br>")
        with gr.Accordion("📚 Additional Information", open=False):
            gr.Markdown("<hr>")
            for text in ADDITIONAL_INFORMATION:
                gr.Markdown(text)

        with gr.Accordion("🔢 Score Calculation Details", open=False):
            gr.Markdown("<hr>")
            for text in SCORE_CALCULATION:
                gr.Markdown(text)

        with gr.Accordion("🦁 AI Singapore & SEA-LION", open=False):
            gr.Markdown("<hr>")
            for text in ABOUT_AISG:
                gr.Markdown(text)

    return demo


# Example usage and testing
if __name__ == "__main__":
    # Your existing mock results...
    config = load_config("leaderboard_v2/config.yaml")
    results = process_runs(config)

    demo = create_gradio_app(results)
    demo.launch(
        share=True,
        favicon_path = "leaderboard_v2/constants/favicon.ico",
        )