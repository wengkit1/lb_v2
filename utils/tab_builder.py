from typing import List, Union, Callable, Dict, Any, Optional
import gradio as gr
from pandas.core.interchange.dataframe_protocol import DataFrame


class TabBuilder:
    """Tab builder that builds list of tab which are tab builders or tab functions"""

    def __init__(self, tabs: List[Union[Callable, 'TabBuilder']],
                 shared_state: Dict[str, Any] = None, tab_name: Optional[str] = None):
        """
        Args:
            tabs: List of tab callables or other TabBuilder objects
            shared_state: Optional shared state dict that gets passed to all tabs
        """
        self.tabs = tabs
        self.shared_state = shared_state or {}
        self.tab_name = tab_name

    def build(self, data: Optional[dict[str, Any]] = None) -> None:
        """Build all tabs in list with optional shared_state"""

        if self.tab_name:
            with gr.Tab(self.tab_name):
                self._build_tabs(data)
        else:
            self._build_tabs(data)

    def _build_tabs(self, data: Optional[dict[str, Any]] = None) -> None:
        """Build the inner tabs structure"""
        with gr.Tabs():
            for tab in self.tabs:
                if isinstance(tab, TabBuilder):
                    tab.shared_state = self.shared_state
                    tab.build(data)
                elif callable(tab):
                    import inspect
                    sig = inspect.signature(tab)
                    params = list(sig.parameters.keys())

                    if len(params) >= 2:
                        tab(data, self.shared_state)
                    else:
                        tab(data)