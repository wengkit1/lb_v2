from typing import Dict, List, Optional
import json
import pandas as pd
import numpy as np
import logging
import os
import glob
from fnmatch import fnmatch
from ..constants.constants import (
    EnConfig, BhasaConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def extract_step_id(dirname):
    """Extract step ID from directory name with multiple patterns"""
    step_id = None

    if 'ba' in dirname:
        # Handle both *ba20000 and xxx_ba20000 formats
        step_id = int(dirname.split('ba')[1].split('_')[0]) if 'ba' in dirname else None
    elif 'step=' in dirname:
        # Handle step= followed by number: step=1000, prefix_step=2000, etc.
        import re
        # Look for 'step=' followed by digits
        match = re.search(r'step=(\d+)', dirname)
        if match:
            step_id = int(match.group(1))

    return step_id if step_id is not None else 0

def find_result_files(root_dir, conditions:List[tuple] = None, identifier:str=None):
    """
    Find all bhasa result files based on path conditions.

    Args:
        root_dir (str): Root directory to start the search
        conditions (list of tuple): List of (level, include, exclude) tuples where:
            - level (int): Path level below root_dir (1-based)
            - include (str): String that must be in the path component
            - exclude (str): String that must not be in the path component

    Returns:
        list: List of paths to matching result files
    """
    print("\n=== Starting find_result_files ===")
    print(f"Root directory: {root_dir}")
    print(f"Search Conditions: {conditions}")

    if not conditions:
        files = []
        for f in glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True):
            if identifier in f:
                files.append(f)
        return files

    root_dir = os.path.abspath(root_dir)
    shortlisted = [root_dir]

    # Process each level
    for include, exclude in sorted(conditions, key=lambda x: x[0]):

        next_shortlist = []

        for path in shortlisted:
            # Get all immediate subdirs/files
            contents = glob.glob(os.path.join(path, "*"))
            # Filter based on conditions
            for item in contents:
                item_name = os.path.basename(item).lower()
                if include and not fnmatch(item_name, include.lower()):
                    continue
                if exclude and fnmatch(item_name, exclude.lower()):
                    continue
                next_shortlist.append(item)
                print(f"  Found: {item}")

        if not next_shortlist:
            print("Warning: No paths matched at this level")
            return []
        shortlisted = next_shortlist

    # Get all json files under final shortlisted paths
    result_files = []
    for path in shortlisted:
        # If last specified level in 'Conditions' is already a JSON file, add it directly
        if os.path.isfile(path) and path.endswith('.json'):
            result_files.append(path)
        # If last specified level in 'Conditions' is a directory, recursively find all JSON files under it
        else:
            json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
            for f in json_files:
                if identifier in f:
                    result_files.append(f)

    if not result_files:
        print("Warning: No JSON files found in shortlisted paths")
    else:
        print(f"\nFound {len(result_files)} JSON files:")
        for f in result_files:
            print(f"  {f}")

    return result_files

def en_reports(result_files: List[str], job_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Process English evaluation results and generate hierarchical reports.

    Args:
        result_files (list): List of paths to JSON result files
        job_name (str, optional): If provided, overwrites the job name extracted from path

    Returns:
        dict: Dictionary containing three DataFrames:
            - 'lang': Overall average performance
            - 'competency': High-level competency results
            - 'task': Detailed task-level results
    """
    # Initialize data storage
    competency_results = {}
    task_results = {}

    # Get the maximum number of digits from all step IDs
    all_step_ids = []
    for f in result_files:
        dirname = os.path.basename(os.path.dirname(f))
        if 'ba' in dirname:
            # Handle both *ba20000 and xxx_ba20000 formats
            step_id = int(dirname.split('ba')[1].split('_')[0]) if 'ba' in dirname else None
            all_step_ids.append(int(step_id))

    n_digits = len(str(max(all_step_ids))) if all_step_ids else 5

    # Process each result file
    for result_file in result_files:
        # Get job_name and step_id from path components
        path_parts = result_file.split(os.sep)
        # Use provided job_name if available, otherwise extract from path
        current_job_name = job_name if job_name is not None else path_parts[-3]
        dirname = os.path.basename(os.path.dirname(result_file))

        step_id = extract_step_id(dirname)
        step_id = str(int(step_id)).zfill(n_digits)   # e.g. 02357
        model_name = f"{current_job_name}_{step_id}"

        with open(result_file) as f:
            data = json.load(f)

        competency_scores = {}
        task_scores = {}

        for task in EnConfig.TASKS:
            norm_raw = 0
            count_raw = 0
            norm = 0
            count = 0

            if task in ["leaderboard_mmlu_pro"]:
                for key in data["results"].keys():
                    if task in key:
                        if "acc,none" in data["results"][key]:
                            # Add to task scores
                            task_name = f"en_{key.replace(f'{task}_', '')}"
                            task_scores[task_name] = data["results"][key]["acc,none"] * 100
                            # Add to competency scores
                            n_options = 10
                            norm_raw += data["results"][key]["acc,none"] * data["n-samples"][key]["effective"]
                            count_raw += data["n-samples"][key]["effective"]
                            norm += ((data["results"][key]["acc,none"] - 1/n_options) * n_options/9) * data["n-samples"][key]["effective"]
                            count = count_raw

            if task in ["leaderboard_gpqa"]:
                for key in data["results"].keys():
                    if task in key and key!=task:
                        if "acc_norm,none" in data["results"][key]:
                            # Add to task scores
                            task_name = f"en_{key.replace(f'{task}_', '')}"
                            n_options = EnConfig.CHOICE_COUNTS[task][key]
                            task_scores[task_name] = data["results"][key]["acc_norm,none"] * 100
                            # Add to competency scores
                            norm_raw += data["results"][key]["acc_norm,none"] * data["n-samples"][key]["effective"]
                            count_raw += data["n-samples"][key]["effective"]
                            norm += (np.clip(data["results"][key]["acc_norm,none"] - 1/n_options, 0, None) * n_options/(n_options-1)) * data["n-samples"][key]["effective"]
                            count = count_raw

            if task in ["leaderboard_bbh", "leaderboard_musr"]:
                for key in data["results"].keys():
                    if task in key and key!=task:
                        if "acc_norm,none" in data["results"][key]:
                            # Add to task scores
                            task_name = f"en_{key.replace(f'{task}_', '')}"
                            n_options = EnConfig.CHOICE_COUNTS[task][key]
                            task_scores[task_name] = data["results"][key]["acc_norm,none"] * 100
                            # Add to competency scores
                            norm_raw += data["results"][key]["acc_norm,none"] * data["n-samples"][key]["effective"]
                            count_raw += data["n-samples"][key]["effective"]
                            norm += (np.clip(data["results"][key]["acc_norm,none"] - 1/n_options, 0, None) * n_options/(n_options-1))
                            count += 1

            if task in ["leaderboard_math_hard"]:
                for key in data["results"].keys():
                    if "leaderboard_math" in key and key!=task:
                        if "exact_match,none" in data["results"][key]:
                            # Add to task scores
                            task_name = f"en_{key.replace('leaderboard_math_', '')}"
                            task_scores[task_name] = data["results"][key]["exact_match,none"] * 100
                            # Add to competency scores
                            norm_raw += data["results"][key]["exact_match,none"] * data["n-samples"][key]["effective"]
                            count_raw += data["n-samples"][key]["effective"]
                            norm = norm_raw
                            count = count_raw

            if task in ["leaderboard_ifeval"]:
                for key in data["results"].keys():
                    if task in key:
                        for k in data["results"][key].keys():
                            if "strict_acc,none" in k:
                                # Add to task scores
                                task_name = f"en_{key.replace(f'{task}_', '')}"
                                task_scores[task_name] = data["results"][key][k] * 100
                                # Add to competency scores
                                norm_raw += data["results"][key][k] * data["n-samples"][key]["effective"]
                                count_raw += data["n-samples"][key]["effective"]
                                norm += data["results"][key][k]
                                count +=1

            if count_raw != 0:
                name = EnConfig.OUTPUT_NAMES[task]
                competency_scores[name] = norm/count*100

        competency_scores["en_total"] = sum(competency_scores.values())/len(competency_scores)

        competency_results[model_name] = competency_scores
        task_results[model_name] = task_scores

    # Create reports
    reports = {}

    # 1. Language report (overall average)
    reports['lang'] = pd.DataFrame(
        {model: {'en': scores['en_total']} for model, scores in competency_results.items()}
    ).T
    reports['lang'].index.name = 'model'

    # 2. Competency report
    columns_order = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
    reports['competency'] = pd.DataFrame(competency_results).T[columns_order]
    reports['competency'].index.name = 'model'

    # 3. Task report
    reports['task'] = pd.DataFrame(task_results).T
    reports['task'].index.name = 'model'



    return reports


def bhasa_reports(result_files: List[str], job_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Process Bhasa evaluation results and directly generate hierarchical reports.

    Args:
        result_files (list): List of paths to JSON result files
        job_name (str, optional): If provided, overwrites the job name extracted from path

    Returns:
        dict: Dictionary containing three DataFrames:
            - 'lang': Overall language performance summary
            - 'competency': Competency-level results
            - 'task': Detailed task-level results
    """
    # Initialize DataFrames data
    lang_data = []
    competency_data = []
    task_data = []

    # Get the maximum number of digits from all step IDs
    all_step_ids = []
    for f in result_files:
        dirname = os.path.basename(os.path.dirname(f))
        if 'ba' in dirname:
            # Handle both *ba20000 and xxx_ba20000 formats
            step_id = int(dirname.split('ba')[1].split('_')[0]) if 'ba' in dirname else None
            all_step_ids.append(int(step_id))

    n_digits = len(str(max(all_step_ids))) if all_step_ids else 5

    # Process each result file
    print(len(result_files))
    for result_file in result_files:
        # Get job_name and step_id from path components
        path_parts = result_file.split(os.sep)
        # Use provided job_name if available, otherwise extract from path
        current_job_name = job_name if job_name is not None else path_parts[-3]
        dirname = os.path.basename(os.path.dirname(result_file))

        step_id = extract_step_id(dirname)
        step_id = str(int(step_id)).zfill(n_digits)   # e.g. 02357
        model_name = f"{current_job_name}_{step_id}"


        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process each language's results
        for lang, competencies in data.items():
            if lang in BhasaConfig.SKIP_LANGUAGES or lang in ["total", "evaluated_as_base"]:
                continue

            for competency_name, tasks in competencies.items():
                # gathers aggregated score for the language
                if competency_name == "total":
                    # Example: if tasks = 85.5 -> score = 85.5
                    # Example: if tasks = {"overall_score": 85.5} -> score = 85.5
                    # Example: if tasks = {"total_score": 85.5} -> score = 85.5
                    score = tasks if isinstance(tasks, (int, float)) else tasks.get('overall_score', tasks.get('total_score', 0))
                    lang_data.append({
                        'model': model_name,
                        'language': lang,
                        'score': score
                    })
                    continue

                # gathers aggregated score for each competency
                if "total" in tasks:
                    total_value = tasks['total']
                    # Handle case where tasks['total'] is a float
                    if isinstance(total_value, (int, float)):
                        score = total_value
                    else:
                        # Example: if total_value = {"overall_score": 85.5} -> score = 85.5
                        # Example: if total_value = {"total_score": 85.5} -> score = 85.5
                        # Example: if total_value = {"other_key": 85.5} -> score = 0
                        score = total_value.get('overall_score', total_value.get('total_score', 0))
                    competency_data.append({
                        'model': model_name,
                        'language': lang,
                        'competency': competency_name,
                        'score': score
                    })

                for task_name, results in tasks.items():
                    # gathers scores for each task
                    if (task_name != "total" and isinstance(results, dict) and
                            BhasaConfig.SCORE_NAMES[0] in results):
                        task_data.append({
                            'model': model_name,
                            'language': lang,
                            'task': task_name,
                            'score': results[BhasaConfig.SCORE_NAMES[0]]
                        })

    # Generate reports
    reports = {}

    # 1. Language summary report
    if lang_data:
        df_lang = pd.DataFrame(lang_data)
        df_lang = pd.pivot_table(df_lang, values='score', index='model', columns='language')
        # aggregate among all languages
        df_lang["sea_total"] = df_lang.mean(axis=1)
        # Reorder columns to put sea_total first, followed by individual language scores
        df_lang = df_lang[["sea_total"] + [col for col in df_lang.columns if col != "sea_total"]]
        reports['lang'] = df_lang

    # 2. Competency report
    if competency_data:
        df_comp = pd.DataFrame(competency_data)
        df_comp['col_name'] = df_comp.apply(lambda x: f"{x['language']}_{x['competency']}", axis=1)
        # Create pivot table with models as rows and language_competency pairs as columns
        # e.g. 'id_nlu', 'th_nlg', etc. with corresponding scores as values
        df_comp = pd.pivot_table(df_comp, values='score', index='model', columns='col_name')
        reports['competency'] = df_comp

    # 3. Task report
    if task_data:
        df_task = pd.DataFrame(task_data)
        df_task['col_name'] = df_task.apply(lambda x: f"{x['language']}_{x['task']}", axis=1)
        df_task = pd.pivot_table(df_task, values='score', index='model', columns='col_name')
        df_task = df_task.reindex(sorted(df_task.columns), axis=1)
        reports['task'] = df_task

    return reports

def save_reports(reports, output_dir=None):
    """
    Save hierarchical reports to CSV files.

    Args:
        reports (dict): Dictionary containing three DataFrames:
            - 'lang': Overall language performance summary
            - 'competency': Competency-level results for each language
            - 'task': Detailed task-level results for each language
        output_dir (str, optional): Directory to save the CSV files.
            If None, uses current working directory

    Returns:
        dict: Dictionary of saved file paths
    """
    # Use current working directory if output_dir is not specified
    if output_dir is None:
        output_dir = os.getcwd()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = {}

    # Save each report
    for report_name, df in reports.items():
        filename = f"{report_name}_report.csv"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath)
        saved_paths[report_name] = filepath

        print(f"\nSaved {report_name} report to: {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

    return saved_paths

def aggregate(
    result_files: List[str],
    eval_type: str = "en",
    job_name: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate and save hierarchical reports from evaluation result files.

    Args:
        result_files: List of paths to JSON result files
        eval_type: Type of evaluation, either "en" or "bhasa"
        job_name: Optional job name to override path-extracted names

    Returns:
        Dictionary containing the generated reports
    """
    if eval_type not in ["en", "bhasa"]:
        raise ValueError('eval_type must be either "en" or "bhasa"')

    # Process results based on type
    if eval_type == "en":
        if not result_files:
            return {}
        reports = en_reports(result_files, job_name=job_name)
    else:  # bhasa
        if not result_files:
            return {}
        reports = bhasa_reports(result_files, job_name=job_name)

    # Round all values to 2 decimal places
    for report_name, df in reports.items():
        reports[report_name] = df.round(2)

    return reports

def merge_reports(bhasa_reports: Dict[str, pd.DataFrame], en_reports: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Merge Bhasa and English reports, handling empty or None inputs."""
    merged_reports = {}

    # Initialize empty reports if None
    bhasa_reports = bhasa_reports or {}
    en_reports = en_reports or {}

    # Default empty DataFrames
    empty_bhasa_lang = pd.DataFrame(columns=['sea_total', 'id', 'th', 'vi'])
    empty_en_lang = pd.DataFrame(columns=['en'])

    # Merge language reports
    bhasa_lang = bhasa_reports.get('lang', empty_bhasa_lang).copy()
    en_lang = en_reports.get('lang', empty_en_lang).copy()
    merged_reports['lang'] = pd.concat([bhasa_lang, en_lang], axis=1).fillna(0)
    merged_reports['lang']['total'] = (
                (merged_reports['lang'].get('sea_total', 0) + merged_reports['lang'].get('en', 0)) / 2).round(2)

    # Merge competency reports
    bhasa_comp = bhasa_reports.get('competency', pd.DataFrame()).copy()
    en_comp = en_reports.get('competency', pd.DataFrame()).copy()
    merged_reports['competency'] = pd.concat([bhasa_comp, en_comp], axis=1).fillna(0)

    # Merge task reports
    bhasa_task = bhasa_reports.get('task', pd.DataFrame()).copy()
    en_task = en_reports.get('task', pd.DataFrame()).copy()
    merged_reports['task'] = pd.concat([bhasa_task, en_task], axis=1).fillna(0)

    # Ensure all DataFrames have 'model' as index name
    for df in merged_reports.values():
        df.index.name = 'model'

    return merged_reports

def main():
    """Main execution function."""


    # ######################################################## g2-9B basemodel
    bhasa_root_dir = en_root_dir = "reports/results/gemma-3-1b-pt"
    job_name = output_dir = "g2_base"
    bhasa_conditions = [
                 (1, "*seahelm*", None)]
    en_conditions = [
                 (1, "lm-eval*", None)]

    # # ######################################################## l3.1-8B basemodel

    # obtain result files
    bhasa_files = find_result_files(bhasa_root_dir, bhasa_conditions)
    en_files = find_result_files(en_root_dir, en_conditions)

    # apply aggregation, get reports, a dictionary of hierarchical_level : dataframe
    bhasa_reports = aggregate(bhasa_files, "bhasa", job_name=job_name)
    print(bhasa_reports)
    print("Bhasa aggregation completed")
    en_reports = aggregate(en_files, "en", job_name=job_name)
    print("English aggregation completed")

    # merge aggregated reports and save out as csv
    merged_reports = merge_reports(bhasa_reports, {})
    save_reports(merged_reports, output_dir)
    print("Merged reports saved")

if __name__ == '__main__':
    main()
