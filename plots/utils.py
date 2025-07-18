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


def model_selector(results: Dict, shared_state: Dict):
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
    print(f"After dropdown creation (normal) - dropdown.choices: {model_choice_dropdown.choices[:3]}")
    selected_models_state = gr.State([])

    shared_state['model_choice_dropdown'] = model_choice_dropdown
    shared_state['selected_models_state'] = selected_models_state
    shared_state['flattened_df'] = flattened_df
    shared_state['grouped_df'] = group_by_experiment(results)


def model_selector_for_experiment(exp_data: Dict, shared_state: Dict):
    """Create model selector components for a single experiment's data"""

    # Get all models from the experiment data
    models = []

    if 'lang' in exp_data and not exp_data['lang'].empty:
        lang_df = exp_data['lang'].reset_index()
        models = lang_df['model'].tolist()

    # If no models from lang, try competency
    elif 'competency' in exp_data and not exp_data['competency'].empty:
        comp_df = exp_data['competency'].reset_index()
        models = comp_df['model'].tolist()

    # If still no models, try task
    elif 'task' in exp_data and not exp_data['task'].empty:
        task_df = exp_data['task'].reset_index()
        models = task_df['model'].tolist()


    # Find baseline models
    baseline_models = [model for model in models if model.startswith('BASELINE_')]
    default_model = baseline_models[0] if baseline_models else (models[0] if models else "")

    # Create dropdown
    model_choice_dropdown = gr.Dropdown(
        choices=models,  # This should be a list of strings
        value=default_model,
        interactive=True,
        multiselect=False,
        container=False,
        label="Select baseline model for delta comparison"
    )
    print(f"After dropdown creation - dropdown.choices: {model_choice_dropdown.choices[:3]}")
    selected_model_state = gr.State(default_model)

    shared_state['model_choice_dropdown'] = model_choice_dropdown
    shared_state['selected_model_state'] = selected_model_state
    shared_state['models'] = models
    shared_state['baseline_models'] = baseline_models