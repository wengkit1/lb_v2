# Developer README - SEA HELM Leaderboard v2

## TabBuilder Architecture

The leaderboard uses a `TabBuilder` class to create modular, composable tab structures. This allows for flexible UI construction and shared state management across tabs.

## Core Concepts

### TabBuilder Class

```python
class TabBuilder:
    def __init__(self, tabs: List[Union[Callable, 'TabBuilder']], 
                 data: Optional[dict] = None,
                 shared_state: Dict[str, Any] = None, 
                 tab_name: Optional[str] = None):
```

- **`tabs`**: List of tab functions or other TabBuilder objects
- **`data`**: Data to pass to all tabs built by the TabBuilder instance (usually experiment results)
- **`shared_state`**: Dictionary for sharing UI components between tabs in the TabBuilder instance
- **`tab_name`**: Optional name to wrap tabs in a named tab

### Tab Function Signature

Tab functions should follow this pattern:

```python
def my_tab_function(exp_data: Dict, shared_state: Dict = None):
    with gr.Tab("Tab Name"):
        # Tab content here
        pass
```

## Usage Patterns

### 1. Simple Tab List

```python
# Create individual tab functions
def overview_tab(exp_data, shared_state=None):
    with gr.Tab("Overview"):
        gr.Markdown("Overview content")

def details_tab(exp_data, shared_state=None):
    with gr.Tab("Details"):
        gr.Markdown("Details content")

# Build tabs
tabs = [overview_tab, details_tab]
builder = TabBuilder(tabs=tabs, data=experiment_data)
builder.build()
```

### 2. Nested Tab Structure

```python
# Create sub-tabs for a main tab
def language_breakdown_tab(df_display, lang_groups, tab_type, is_hero_run):
    language_tab_functions = []
    
    # Add language-specific tabs
    for lang_name, lang_cols in lang_groups.items():
        tab_func = create_single_language_tab(lang_name, lang_cols, df_display)
        language_tab_functions.append(tab_func)
    
    # Add hero line plot if applicable
    if is_hero_run:
        hero_plot_tab = hero_line_plot_tab(df_display, tab_type)
        language_tab_functions.append(hero_plot_tab)
    
    # Build nested structure
    language_builder = TabBuilder(language_tab_functions)
    language_builder.build()
```

### 3. Shared State Management

```python
def comparison_tab(results: Dict):
    def comparison_tab_func(*args):
        with gr.Tab("Comparison"):
            # Initialize shared state
            shared_state = {}
            
            # Create shared components
            model_selector(results, shared_state)
            
            # Define tabs that use shared state
            tabs = [
                comparison_table_tab,
                performance_plot_tab
            ]
            
            # Build with shared state
            builder = TabBuilder(tabs=tabs, shared_state=shared_state)
            builder.build()
    
    return comparison_tab_func
```

### 4. Named Tab Wrapper

```python
# Create a main tab that contains sub-tabs
experiment_tab_structure = [
    delta_comparison_plot_tab,
    table_view,
    pareto_plot_tab,
    contour_plot_tab
]

# Wrap in named tab
experiment_tab = TabBuilder(
    data=cleaned_data,
    tabs=experiment_tab_structure,
    tab_name="My Experiment"
)
```

## Real Examples from Codebase

### Main App Structure

```python
def create_main_tabs(results_dict):
    # Define tab structure for each experiment
    experiment_tab_structure = [
        delta_comparison_plot_tab,
        table_view,
        pareto_plot_tab,
        contour_plot_tab,
        competency_selection_tab
    ]
    
    # Create tabs for each experiment
    experiments = get_experiments_only(results_dict)
    experiment_tabs = []
    
    for exp_name, exp_data in experiments.items():
        cleaned_data = clean_column_names(exp_data)
        experiment_tab = TabBuilder(
            data=cleaned_data,
            tabs=experiment_tab_structure,
            tab_name=exp_name  # Creates tab named after experiment
        )
        experiment_tabs.append(experiment_tab)
    
    # Add comparison tab
    all_tabs = experiment_tabs + [comparison_tab(results_dict)]
    
    # Build main structure
    main_builder = TabBuilder(all_tabs)
    main_builder.build()
```

### Table View with Sub-tabs

```python
def table_view(exp_data: Dict, *args):
    """Main table view tab with overall, competency, and task tabs."""
    table_sub_tabs = [
        overall_tab,
        competency_tab,
        task_tab
    ]
    
    # Build sub-tabs directly (no wrapper needed)
    for tab_func in table_sub_tabs:
        tab_func(exp_data, *args)
```

### Delta Comparison with Shared State

```python
def delta_comparison_plot_tab(exp_data: Dict, shared_state: Dict):
    with gr.Tab("Delta Comparison"):
        # Create local shared state for this tab's components
        local_shared_state = {}
        model_selector_for_experiment(exp_data, local_shared_state)
        
        # Show the model selector dropdown
        local_shared_state.get('model_choice_dropdown')
        
        # Define sub-tabs that will use the shared dropdown
        delta_sub_tabs = [
            language_performance_tab,
            competency_performance_tab,
            task_performance_tab
        ]
        
        # Build with shared state
        delta_builder = TabBuilder(tabs=delta_sub_tabs, shared_state=local_shared_state)
        delta_builder.build(exp_data)
```

## Creating New Tabs

### Step 1: Define Tab Function

```python
def my_new_tab(exp_data: Dict, shared_state: Dict = None):
    """My new tab for displaying custom analysis."""
    with gr.Tab("My Analysis"):
        # Get data
        df = exp_data.get('lang', pd.DataFrame())
        
        if df.empty:
            gr.Markdown("No data available")
            return
        
        # Create UI components
        gr.Markdown("## My Custom Analysis")
        
        # Use shared state if needed
        if shared_state and 'model_choice_dropdown' in shared_state:
            dropdown = shared_state['model_choice_dropdown']
            # Connect dropdown to some functionality
        
        # Add content
        gr.DataFrame(df.reset_index())
```

### Step 2: Add to Tab Structure

```python
# Add to existing structure
experiment_tab_structure = [
    delta_comparison_plot_tab,
    table_view,
    my_new_tab,  # <-- Add here
    pareto_plot_tab,
    contour_plot_tab
]
```

### Step 3: Handle Shared State (if needed)

```python
def my_tab_with_shared_components(exp_data: Dict, shared_state: Dict = None):
    with gr.Tab("My Tab"):
        # Create shared components
        if not shared_state:
            shared_state = {}
        
        # Add components to shared state
        my_dropdown = gr.Dropdown(choices=["A", "B"], label="My Dropdown")
        shared_state['my_dropdown'] = my_dropdown
        
        # Create sub-tabs that use the dropdown
        sub_tabs = [
            lambda data, state: subtab_1(data, state),
            lambda data, state: subtab_2(data, state)
        ]
        
        builder = TabBuilder(tabs=sub_tabs, shared_state=shared_state)
        builder.build(exp_data)
```

## Best Practices

### 1. Data Flow

```python
# Good: Pass data explicitly
def my_tab(exp_data: Dict, shared_state: Dict = None):
    df = exp_data.get('lang')
    # Use df...

```

### 2. Component Creation

```python
# Create components in the right context
def my_tab(exp_data, shared_state=None):
    with gr.Tab("My Tab"):
        # Components created inside tab context
        dropdown = gr.Dropdown(...)
        plot = gr.Plot(...)

# This will not work, due to gradio's design
dropdown = gr.Dropdown(...)  # Created before tab context
def my_tab(exp_data, shared_state=None):
    with gr.Tab("My Tab"):
        # dropdown already exists, might cause issues
```


## Common Patterns Summary

1. **Simple tabs**: Function list → TabBuilder → build()
2. **Nested tabs**: Create sub-functions → TabBuilder for each level
3. **Shared state**: Initialize components → Store in dict → Pass to tabs
4. **Named tabs**: Use `tab_name` parameter to wrap content
5. **Conditional logic**: Check metadata or data before adding tabs

This architecture makes the UI highly modular and allows for complex, interactive dashboards while keeping the code organized and maintainable.