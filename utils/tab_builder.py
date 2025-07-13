from typing import List, Union, Callable, Dict, Any
import gradio as gr

class TabBuilder:
    """Tab builder that builds list of tab which are tab builders or tab functions"""

    def __init__(self, tabs: List[Union[Callable, 'TabBuilder']]):
        """
        Args:
            tabs: List of tab callables or other TabBuilder objects
        """
        self.tabs = tabs

    def build(self, data: Dict[str, Any]) -> None:
        """Build all tabs according to the structure"""
        with gr.Tabs():
            for tab in self.tabs:
                if isinstance(tab, TabBuilder):
                    # Another TabBuilder - let it handle its own tabs
                    tab.build(data)
                elif callable(tab):
                    # Single tab function
                    tab(data)

    def __call__(self, data: Dict[str, Any]) -> None:
        """Make TabBuilder itself callable as a tab"""
        self.build(data)
