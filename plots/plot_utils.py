from typing import Dict, List
import gradio as gr
import pandas as pd
import re

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


def group_columns_by_language(df: pd.DataFrame):
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


def extract_hyperparameters(model_name: str) -> Dict:
    """Extract hyperparameters from model names."""
    hps = {}

    # Datamix pattern: n4-g8-en_ratio0.1-cc_ratio0.1-code_ratio0.1
    datamix_patterns = [
        r'en_ratio([\d.]+)',
        r'cc_ratio([\d.]+)',
        r'code_ratio([\d.]+)',
        r'hq_ratio([\d.]+)'
    ]

    for pattern in datamix_patterns:
        match = re.search(pattern, model_name)
        if match:
            param_name = pattern.split('(')[0].replace('_ratio', '_ratio')
            hps[param_name] = float(match.group(1))

    # HPS-sweep pattern: hps-sweep_smc_gemma-3-4b-it_SEQLEN8192_MBS4_N4_FULL_SHARD_LR1e-4_GBS1024_DUR20e9tok_LIGER1_WD1e-5
    hps_patterns = [
        (r'LR([\de.-]+)', 'LR'),
        (r'GBS(\d+)', 'GBS'),
    ]

    for pattern, param_name in hps_patterns:
        match = re.search(pattern, model_name)
        if match:
            try:
                if 'e' in match.group(1) or '.' in match.group(1):
                    hps[param_name] = float(match.group(1))
                else:
                    hps[param_name] = int(match.group(1))
            except ValueError:
                hps[param_name] = match.group(1)

    return hps


def get_available_hyperparams(exp_data: Dict) -> List[str]:
    """Extract all unique hyperparameters from the dataframes."""
    combined_df = combine_dataframes(exp_data)
    if combined_df is None or combined_df.empty:
        return []

    # Extract hyperparameters for all models
    combined_df['hyperparams'] = combined_df['model'].apply(extract_hyperparameters)

    all_hp_keys = set()
    for hp_dict in combined_df['hyperparams']:
        all_hp_keys.update(hp_dict.keys())

    return sorted(list(all_hp_keys))


def get_all_metrics(exp_data: Dict) -> List[str]:
    """Get all available metrics from lang, competency, and task dataframes."""
    all_metrics = []

    # From lang dataframe
    if 'lang' in exp_data and not exp_data['lang'].empty:
        lang_cols = [col for col in exp_data['lang'].columns if col != 'model']
        all_metrics.extend(lang_cols)

    # From competency dataframe
    if 'competency' in exp_data and not exp_data['competency'].empty:
        comp_cols = [col for col in exp_data['competency'].columns if col != 'model']
        all_metrics.extend(comp_cols)

    # From task dataframe
    if 'task' in exp_data and not exp_data['task'].empty:
        task_cols = [col for col in exp_data['task'].columns if col != 'model']
        all_metrics.extend(task_cols)

    return sorted(list(set(all_metrics)))


def combine_dataframes(exp_data: Dict) -> pd.DataFrame:
    """Combine all dataframes into one with all metrics."""
    combined_df = None

    for df_name in ['lang', 'competency', 'task']:
        if df_name in exp_data and not exp_data[df_name].empty:
            df = exp_data[df_name].reset_index()
            if 'model' not in df.columns:
                df['model'] = df.index

            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='model', how='outer')

    return combined_df


def model_selector(results: Dict, shared_state: Dict):
    """Create the shared components and store them in shared_state"""

    flattened_df = flatten(results)
    models = flattened_df['model'].tolist() if not flattened_df.empty else []
    models.sort()

    # Create shared components here
    model_choice_dropdown = gr.Dropdown(
        choices=models,
        value='',
        interactive=True,
        multiselect=True,
        container=False,
        allow_custom_value=True,
        label="Select models for comparison (click out of dropdown to apply)",
    )
    selected_models_state = gr.State([])

    shared_state['model_choice_dropdown'] = model_choice_dropdown
    shared_state['selected_models_state'] = selected_models_state
    shared_state['flattened_df'] = flattened_df
    shared_state['grouped_df'] = group_by_experiment(results)


def model_selector_for_experiment(exp_data: Dict, shared_state: Dict):
    """Create model selector components for a single experiment's data"""

    models = []
    df = exp_data[next(iter(exp_data))]
    if isinstance(df, pd.DataFrame) and not df.empty:
        df_reset = df.reset_index()
        if 'model' in df_reset.columns:
            models = df_reset['model'].tolist()
    models.sort()

    baseline_models = [model for model in models if model.startswith('BASELINE_')]
    default_model = baseline_models[0] if baseline_models else (models[0] if models else "")

    # Create dropdown
    model_choice_dropdown = gr.Dropdown(
        choices=models,
        value=default_model,
        interactive=True,
        multiselect=False,
        container=False,
        label="Select baseline model for delta comparison"
    )
    selected_model_state = gr.State(default_model)

    shared_state['model_choice_dropdown'] = model_choice_dropdown
    shared_state['selected_model_state'] = selected_model_state
    shared_state['models'] = models
    shared_state['baseline_models'] = baseline_models