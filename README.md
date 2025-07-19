# SEA HELM Leaderboard v2

A comprehensive evaluation pipeline for processing and visualizing language model results across Southeast Asian languages and English benchmarks.

## Overview

This pipeline aggregates evaluation results from multiple experiments and hero runs, generates hierarchical reports, and provides an interactive web interface for comparing model performance across different languages and tasks.

## Project Structure

```
leaderboard_v2/
├── main.py                    # Main entry point
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── constants/                # Configuration constants
│   ├── constants.py          # Task definitions, language mappings
│   ├── text_blobs.py        # UI text content
│   └── favicon.ico          # Web interface icon
│
├── utils/                    # Core utilities
│   ├── __init__.py
│   ├── aggregate.py         # Data aggregation logic
│   ├── process_config.py    # Configuration processing
│   └── tab_builder.py       # UI tab construction utility
│
├── plots/                    # Visualization components
│   ├── __init__.py
│   ├── plot_utils.py        # Common plotting utilities
│   ├── contour_plot.py      # Hyperparameter contour plots
│   ├── pareto_plot.py       # Pareto front analysis
│   ├── delta_comparison.py  # Model comparison plots
│   ├── competency_selection_view.py  # Competency filtering
│   │
│   ├── comparison_tab/      # Cross-experiment comparison
│   │   ├── comparison_tab.py
│   │   └── performance_plot.py
│   │
│   └── table_tabs/          # Data table views
│       ├── table_view.py
│       └── hero_line_plot.py
│
└── js/                      # JavaScript utilities
    └── dynamic_averages.js  # Dynamic table calculations
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Create or modify `config.yaml` based on your experiment structure:

```yaml
experiments:
  experiment_name:
    path: "path/to/experiment/results"
    # Optional: custom conditions for file filtering
    en_conditions:
      - ["directory_pattern", "exclude_pattern"]
    bhasa_conditions:
      - ["directory_pattern", "exclude_pattern"]
  
  baselines:
    baseline_name:
      path: "path/to/baseline/results"
      target_experiments: [0, 1]  # Indices of experiments to merge with

hero_runs:
  hero_run_name:
    path: "path/to/hero_run/results"
  
  baselines:
    baseline_name:
      path: "path/to/baseline/results"
      target_experiments: [0]

output:
  base_dir: "./reports/results/aggregated_reports"
  save_individual: true
  save_merged: true
```

### 3. Run the Pipeline

**With GUI (default):**
```bash
python -m leaderboard_v2.main --config config.yaml --port 7060
```

**Without GUI (aggregation only):**
```bash
python -m leaderboard_v2.main --config config.yaml --no-gui
```

**Run from project root:**
```bash
python leaderboard_v2/main.py --config leaderboard_v2/config.yaml --port 7060
```

## Configuration Structure

### Experiments vs Hero Runs

- **Experiments**: Regular training runs with multiple configurations
- **Hero Runs**: Special training runs tracked over time (e.g., checkpoints during training)
- **Baselines**: Reference models merged into experiments for comparison

### Directory Structure Requirements

The pipeline expects evaluation results in this structure:

**For Experiments:**
```
experiment_root/
├── experiment_config_1/
│   ├── sub_config/
│   │   ├── *seahelm*.json    # Bhasa evaluation results
│   │   └── *lm-eval*.json    # English evaluation results
│   └── ...
└── experiment_config_2/
    └── ...
```

**For Hero Runs:**
```
hero_run_root/
├── ba1000/                   # Step 1000
│   ├── *seahelm*.json
│   └── *lm-eval*.json
├── ba2000/                   # Step 2000
│   ├── *seahelm*.json
│   └── *lm-eval*.json
└── ...
```

**For Baselines:**
```
baseline_root/
├── some_folder/
│   ├── *seahelm*.json
│   └── *lm-eval*.json
```

### Configuration Options

#### File Filtering Conditions
```yaml
en_conditions:
  - ["directory_pattern", "exclude_pattern"]
  - ["*", null]  # Include all, exclude none
  - ["lm-eval*", null]  # Include dirs starting with "lm-eval"

bhasa_conditions:
  - ["*seahelm*", null]  # Include dirs containing "seahelm"
```

#### Baseline Integration
```yaml
baselines:
  baseline_name:
    path: "path/to/baseline"
    target_experiments: [0, 1]  # Merge into first 2 experiments
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to configuration file | `config.yaml` |
| `--port` | Web interface port | `7060` |
| `--no-gui` | Run aggregation without launching GUI | `False` |

## Output Structure

### Generated Reports

The pipeline creates hierarchical CSV reports:

1. **`lang_report.csv`** - Overall performance by language
2. **`competency_report.csv`** - Performance by competency area  
3. **`task_report.csv`** - Detailed task-level results

### Web Interface Features

- **Experiment Tabs**: Individual experiment results with language breakdowns
- **Comparison Tab**: Cross-experiment model comparison
- **Delta Comparison**: Performance differences against baselines
- **Pareto Plots**: Multi-objective optimization analysis
- **Contour Plots**: Hyperparameter space visualization
- **Hero Run Tracking**: Training progress over time

## Data Structure

### Results Dictionary

```python
results = {
    'experiment_name': {
        'lang': pd.DataFrame,      # Language-level scores
        'competency': pd.DataFrame, # Competency-level scores
        'task': pd.DataFrame,      # Task-level scores
        '_meta': {
            'is_hero_run': False,
            'has_baseline': True
        }
    },
    'hero_run_name': {
        'lang': pd.DataFrame,
        'competency': pd.DataFrame,
        'task': pd.DataFrame,
        '_meta': {
            'is_hero_run': True,
            'has_baseline': False
        }
    },
    'BASELINE_section_name': {
        'lang': pd.DataFrame,
        'competency': pd.DataFrame,
        'task': pd.DataFrame,
        '_meta': {
            'is_baseline': True,
            'baseline_for': [0, 1]
        }
    }
}
```

### Model Naming Conventions

- **Experiments**: `{experiment_name}_{step_id}`
- **Hero Runs**: `{hero_run_name}_{step_id}`
- **Baselines**: `BASELINE_{section}_{baseline_name}_{step_id}`

## Supported Evaluations

### Bhasa (Southeast Asian Languages)
- Indonesian (id)
- Vietnamese (vi)
- Thai (th) 
- Tamil (ta)

### English Benchmarks
- MMLU-PRO
- BBH (Big Bench Hard)
- GPQA
- MATH Level 5
- IFEval
- MUSR

## Dependencies

Key requirements:
- `gradio >= 5.33.0`
- `gradio-leaderboard >= 0.0.13`
- `matplotlib >= 3.10.3`
- `plotly >= 6.1.2`
- `scipy >= 1.15.3`
- `pandas`
- `numpy`
- `pyyaml`

## Example Usage

### Basic Configuration
```yaml
experiments:
  my_experiment:
    path: "/path/to/experiment/results"

output:
  base_dir: "./reports"
  save_individual: true
  save_merged: true
```

### With Baselines and Hero Runs
```yaml
experiments:
  g3_datamix_4b:
    path: "reports/results/g3_datamix_4b"
  
  baselines:
    gemma-3-1b:
      path: "reports/results/gemma-3-1b-pt"
      target_experiments: [0]

hero_runs:
  gemma3_herorun:
    path: "reports/results/g3_4b_herorun_6e-7/herorun_4b"
  
  baselines:
    gemma-3-1b:
      path: "reports/results/gemma-3-1b-pt"
      target_experiments: [0]

output:
  base_dir: "./reports/results/aggregated_reports"
  save_individual: true
  save_merged: true
```

## Features

### Interactive Visualizations
- **Delta Comparison**: Color-coded performance differences
- **Pareto Analysis**: Multi-objective optimization frontiers
- **Contour Plots**: Hyperparameter space exploration
- **Hero Run Tracking**: Training progress visualization

### Data Export
- CSV reports at multiple granularity levels
- Hierarchical organization (language → competency → task)
- Baseline integration for easy comparison

### Flexible Configuration
- Custom file filtering patterns
- Selective baseline integration
- Configurable output options

## Advanced Usage

### Custom File Patterns
```yaml
experiments:
  my_experiment:
    path: "/path/to/results"
    en_conditions:
      - ["step*", null]        # Include dirs starting with "step"
      - ["*", "*debug*"]       # Exclude dirs containing "debug"
    bhasa_conditions:
      - ["checkpoint*", null]  # Include dirs starting with "checkpoint"
```

### Multiple Baselines
```yaml
experiments:
  exp1:
    path: "/path/to/exp1"
  exp2:
    path: "/path/to/exp2"
    
  baselines:
    baseline_a:
      path: "/path/to/baseline_a"
      target_experiments: [0]    # Only exp1
    baseline_b:
      path: "/path/to/baseline_b"
      target_experiments: [0, 1] # Both exp1 and exp2
```
