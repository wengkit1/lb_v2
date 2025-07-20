from utils import TabBuilder
import gradio as gr

def tab_a(*args):
    with gr.Tab("child_a"):
        gr.Markdown("<br>")


def tab_b(*args):
    with gr.Tab("child_b"):
        gr.Markdown("<br>")


with gr.Blocks() as demo:
    gr.Markdown("<br>")
    TabBuilder(tabs=[tab_a, tab_b]).build()


with gr.Blocks() as recursion_demo:
    composed_tab = TabBuilder(tabs=[tab_a, tab_b], tab_name="parent")
    TabBuilder(tabs=[tab_a, tab_b, composed_tab]).build()


def shared_tab_a(data, shared_state):
   with gr.Tab("child_a"):
       gr.Markdown(f"A: {shared_state['shared_state']}")


def shared_tab_b(data, shared_state):
   with gr.Tab("child_b"):
       gr.Markdown(f"B: {shared_state['shared_state']}")


def tab_c(*args, shared_state=None):
    if not shared_state:
        with gr.Tab("child_c"):
            gr.Markdown("I do not share shared_state!")


with gr.Blocks() as shared_state_demo:
   shared_state = {"shared_state": "We are sharing a dict!"}
   sharing_tabs = TabBuilder(tabs=[shared_tab_a, shared_tab_b],
                             shared_state=shared_state)
   # note that not naming the tabbuilder object will create a separate tabs context
   # which would stack them vertically.
   sharing_tabs_named = TabBuilder(tabs=[shared_tab_a, shared_tab_b],
                             shared_state=shared_state, tab_name="shared_tabs")
   TabBuilder(tabs=[tab_c, sharing_tabs_named]).build()


shared_state_demo.launch(
    server_port=7060,
    share=False,
)