import os
from typing import Dict, Optional

import pandas as pd
import yaml

from leaderboard_v2.utils.aggregate import (
    find_result_files,
    aggregate,
    merge_reports,
    save_reports
)
from ..constants.constants import (
    SEAHELM_FILE_IDENTIFIER,
    OPENLLM_FILE_IDENTIFIER
)

def load_config(config_path: str = "leaderboard_v2/config.yaml") -> Dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def process_runs(config: Optional[Dict] = None) -> Dict:
    """
    Process both experiments and hero runs based on configuration.

    Args:
        config: Dictionary containing experiment configuration settings.
              If None, a dummy configuration will be used.

    Returns:
        Dict: Results dictionary with processed experiments and hero runs
    """
    results = {}

    # Process regular experiments
    experiments_config = config.get('experiments', {})
    if experiments_config:
        experiment_results = process_run_section(
            experiments_config,
            section_name="experiments",
            is_hero_run=False
        )
        results.update(experiment_results)

    # Process hero runs
    hero_runs_config = config.get('hero_runs', {})
    if hero_runs_config:
        hero_results = process_run_section(
            hero_runs_config,
            section_name="hero_runs",
            is_hero_run=True
        )
        results.update(hero_results)

    # Save all results
    output_dir = config.get('output', {}).get('base_dir', 'reports')
    for experiment_name, experiment_data in results.items():
        if not experiment_name.startswith('BASELINE_'):  # Skip saving baselines separately
            data_to_save = {k: v for k, v in experiment_data.items() if k != '_meta'}
            save_reports(data_to_save, os.path.join(output_dir, experiment_name))

    return results


def process_run_section(run_config: Dict, section_name: str, is_hero_run: bool = False) -> Dict:
    """
    Process a section of runs (either experiments or hero_runs).

    Args:
        run_config: Configuration for this section
        section_name: Name of the section ("experiments" or "hero_runs")
        is_hero_run: Whether these are hero runs

    Returns:
        Dict: Processed results for this section
    """
    results = {}
    baselines_config = run_config.get('baselines', {})

    # Process main runs first
    for run_name, settings in run_config.items():
        if run_name == "baselines":
            continue

        run_path = settings.get('path')

        # Process bhasa and en results
        bhasa_df = aggregate(
            find_result_files(run_path, settings.get('bhasa_conditions'), SEAHELM_FILE_IDENTIFIER),
            "bhasa"
        )
        en_df = aggregate(
            find_result_files(run_path, settings.get('en_conditions'), OPENLLM_FILE_IDENTIFIER),
            "en"
        )

        # Store with hero_run flag
        run_result = merge_reports(bhasa_df, en_df)
        run_result['_meta'] = {'is_hero_run': is_hero_run}
        results[run_name] = run_result

    # Process baselines for this section
    if baselines_config:
        baseline_results = process_baselines(baselines_config, results, section_name)
        results.update(baseline_results)

    return results


def process_baselines(baselines_config: Dict, main_results: Dict, section_name: str) -> Dict:
    """
    Process baselines and add them to target experiments.

    Args:
        baselines_config: Baseline configuration
        main_results: Main experiment results to merge baselines into
        section_name: Section name for baseline identification
        is_hero_run: Whether this is for hero runs

    Returns:
        Dict: Updated results with baselines merged
    """
    updated_results = {}

    for baseline_name, baseline_settings in baselines_config.items():
        baseline_path = baseline_settings.get('path')
        target_experiments = baseline_settings.get('target_experiments', [])

        # Create unique baseline identifier
        baseline_key = f"BASELINE_{section_name}_{baseline_name}"

        # Process baseline data
        bhasa_df = aggregate(
            find_result_files(baseline_path, None, SEAHELM_FILE_IDENTIFIER),
            "bhasa",
            job_name=baseline_key,
        )
        en_df = aggregate(
            find_result_files(baseline_path, None, OPENLLM_FILE_IDENTIFIER),
            "en",
            job_name=baseline_key,
        )

        baseline_data = merge_reports(bhasa_df, en_df)
        baseline_data['_meta'] = {
            'is_baseline': True,
            'baseline_for': target_experiments
        }

        # Get list of main experiments
        main_experiment_keys = [k for k in main_results.keys() if not k.startswith('BASELINE_')]

        # Merge baseline into target experiments
        for target_idx in target_experiments:
            if target_idx < len(main_experiment_keys):
                experiment_key = main_experiment_keys[target_idx]
                merged_experiment = merge_baseline_with_experiment(
                    main_results[experiment_key],
                    baseline_data,
                )
                updated_results[experiment_key] = merged_experiment

        # Also store the baseline separately for reference
        updated_results[baseline_key] = baseline_data

    return updated_results


def merge_baseline_with_experiment(experiment_data: Dict, baseline_data: Dict) -> Dict:
    """
    Merge baseline data with experiment data.

    Args:
        experiment_data: Main experiment data
        baseline_data: Baseline data to merge
        is_hero_run: Whether this is a hero run

    Returns:
        Dict: Merged experiment data
    """
    merged_experiment = {}

    for key in experiment_data.keys():
        if key == '_meta':
            # Preserve metadata and add baseline info
            merged_experiment[key] = experiment_data[key].copy()
            merged_experiment[key]['has_baseline'] = True
            continue

        experiment_df = experiment_data[key]
        baseline_df = baseline_data[key]
        merged_df = pd.concat([experiment_df, baseline_df], join='outer').fillna(0)
        merged_experiment[key] = merged_df

    return merged_experiment


def is_baseline_model(model_name: str) -> bool:
    """
    Check if a model name represents a baseline.

    Args:
        model_name: Model name to check

    Returns:
        bool: True if this is a baseline model
    """
    return model_name.startswith('BASELINE_')


def get_baseline_info(model_name: str) -> Dict:
    """
    Extract baseline information from model name.

    Args:
        model_name: Baseline model name

    Returns:
        Dict: Baseline information including original name
    """
    if not is_baseline_model(model_name):
        return {}

    # Parse BASELINE_section_name format
    parts = model_name.split('_', 2)  # Split into max 3 parts
    if len(parts) >= 3:
        return {
            'baseline_name': parts[2],
            'is_baseline': True
        }

    return {'is_baseline': True}

if __name__ == "__main__":
    config = load_config()
    results = process_runs(config)
    print(results['g3_datamix_4b']['_meta'])