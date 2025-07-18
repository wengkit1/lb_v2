from typing import List, Union, Callable, Dict, Any, Optional
import gradio as gr


class TabBuilder:
    """Tab builder that builds a list of tabs which are tab builders or tab functions"""

    def __init__(self, tabs: List[Union[Callable, 'TabBuilder']],
                 data: Optional[dict[str, Any]] = None,
                 shared_state: Dict[str, Any] = None, tab_name: Optional[str] = None):
        """
        Args:
            tabs: List of tab callables or other TabBuilder objects
            shared_state: Optional shared state dict that gets passed to all tabs
        """
        self.tabs = tabs
        self.data = data
        self.shared_state = shared_state or {}
        self.tab_name = tab_name

    def build(self, override_data: Optional[dict[str, Any]] = None) -> None:
        """Build all tabs in list with optional shared_state"""
        data_to_use = override_data if override_data is not None else self.data
        if self.tab_name:
            with gr.Tab(self.tab_name):
                self._build_tabs(data_to_use)
        else:
            self._build_tabs(data_to_use)

    def _build_tabs(self, data: Optional[dict[str, Any]] = None) -> None:
        """Build the inner tabs structure"""
        with gr.Tabs():
            for tab in self.tabs:
                if isinstance(tab, TabBuilder):
                    merged_state = {**self.shared_state, **tab.shared_state}
                    tab.shared_state = merged_state
                    tab.build(data)

                elif callable(tab):
                    tab(data, self.shared_state)