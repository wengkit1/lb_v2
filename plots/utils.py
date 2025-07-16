from typing import Dict
import pandas as pd
import gradio as gr

def flatten(results: Dict):
    all_models = {}

    for experiment_key, experiment_value in results.items():
        if experiment_key.startswith('BASELINE_'):
            continue

        experiment_data = experiment_value
        lang_df = experiment_data.get('lang', pd.DataFrame())

        if lang_df.empty:
            continue

        df_reset = lang_df.reset_index()
        for _, row in df_reset.iterrows():
            model_name = row['model']
            all_models[model_name] = row.to_dict()

    if not all_models:
        gr.Markdown("No model data available for comparison.")
        return

    flattened_df = pd.DataFrame(list(all_models.values()))
    return flattened_df


def group_by_experiment(results: Dict) -> pd.DataFrame:
    """Create a dataframe with model family information from all experiments"""
    all_models = {}

    for experiment_key, experiment_value in results.items():
        if experiment_key.startswith('BASELINE_'):
            continue

        experiment_data = experiment_value
        lang_df = experiment_data.get('lang', pd.DataFrame())
        if lang_df.empty:
            continue

        df_reset = lang_df.reset_index()

        for _, row in df_reset.iterrows():
            model_name = row['model']
            model_data = row.to_dict()
            model_data['model_family'] = experiment_key
            all_models[model_name] = model_data

    if all_models:
        return pd.DataFrame(list(all_models.values()))
    else:
        return pd.DataFrame()


def create_shared_components(results: Dict, shared_state: Dict):
    """Create the shared components and store them in shared_state"""

    flattened_df = flatten(results)
    models = flattened_df['model'].tolist() if not flattened_df.empty else []
    models.sort()

    # Create shared components here
    model_choice_dropdown = gr.Dropdown(
        choices=models,
        value=[],
        interactive=True,
        multiselect=True,
        container=False,
        label="Select models for comparison (click out of dropdown to apply)"
    )

    selected_models_state = gr.State([])

    shared_state['model_choice_dropdown'] = model_choice_dropdown
    shared_state['selected_models_state'] = selected_models_state
    shared_state['flattened_df'] = flattened_df
    shared_state['grouped_df'] = group_by_experiment(results)