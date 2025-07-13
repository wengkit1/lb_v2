from pandas import DataFrame

from leaderboard_v2.constants.text_blobs import ABOUT_SEAHELM, ADDITIONAL_INFORMATION, ABOUT_AISG, SCORE_CALCULATION
from leaderboard_v2.plots.table_view import table_view
from leaderboard_v2.utils import process_runs, load_config, TabBuilder

from .constants.constants import LANGUAGE_NAMES
from .plots.delta_comparison import delta_comparison_tab
import gradio as gr
slider_css = """
        .gradio-container .form {
            overflow-y: auto !important;
            max-height: 800px !important;
        }
        """


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
                        # Replace language code with full name
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


def create_experiment_tab_function(exp_name, exp_data, tab_structure):
    """Factory function that returns a tab function for a specific experiment"""
    def experiment_tab_func(unused):
        with gr.Tab(exp_name):
            filtered_data = {k: v for k, v in list(exp_data.items())[:-1]}
            cleaned_data = clean_column_names(filtered_data)

            builder = TabBuilder(tab_structure)
            builder.build(cleaned_data)

    return experiment_tab_func


def create_main_tabs(results_dict: dict[str, dict[str, DataFrame]],):
    experiment_tab_structure = [
        delta_comparison_tab,
        table_view,
    ]
    experiments = get_experiments_only(results_dict)
    experiment_tabs = [
        create_experiment_tab_function(exp_name, exp_data, experiment_tab_structure)
        for exp_name, exp_data in experiments.items()
    ]
    main_builder = TabBuilder(experiment_tabs)
    main_builder.build({})

def create_gradio_app(results_dict):
    """Main entry point - create the Gradio app"""
    with gr.Blocks(css=slider_css) as demo:
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