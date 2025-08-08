from agents.conversation_agent import run_conversation_graph
import gradio as gr


class TripPlannerUI:
    def __init__(self):
        self.run_conversation_graph = run_conversation_graph
        self.context = {}

    def agent_chat(self, prompt, chat_history):
        state = self.run_conversation_graph(prompt, chat_history, self.context)
        if isinstance(state, dict) and 'context' in state and state['context']:
            self.context = state['context']
        return state

    def respond(self, message, chat_history):
        chat_history = chat_history or []
        # Normalize history structure to list of dicts
        if chat_history and isinstance(chat_history[0], (list, tuple)):
            normalized = []
            for user, agent in chat_history:
                normalized.append({"role": "user", "content": user})
                normalized.append({"role": "assistant", "content": agent})
            chat_history = normalized

        state = self.agent_chat(message, chat_history)
        chat_history.append({"role": "user", "content": message})
        if isinstance(state, dict):
            resp_text = state.get('response', '')
            missing_info = state.get('missing_info', False)
        else:
            resp_text = str(state)
            missing_info = False
        chat_history.append({"role": "assistant", "content": resp_text})
        return "", chat_history

    def launch(self):
        css = """
        <style>
        /* Remove footer */
        footer, .svelte-1ipelgc {
            display: none !important;
        }

        /* Remove top/bottom padding and center content */
        .gradio-container {
            padding: 0 !important;
            margin: 0 !important;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Let chatbot grow naturally */
        .gr-chatbot {
            flex-grow: 1;
            min-height: 100px !important;
            height: auto !important;
            max-height: none !important;
            overflow-y: auto !important;
        }

        /* Remove weird margins between elements */
        main > div {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        /* Input alignment */
        .message-input {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        </style>
        """

        with gr.Blocks(theme=gr.themes.Soft(), title="Trip Planner AI") as demo:
            gr.HTML(css)
            gr.Markdown("""# 🌍 Trip Planner AI\nPlan your adventure: ask for destinations, flights, activities, weather, budget.""")
            chatbot = gr.Chatbot(show_label=False, type='messages')
            with gr.Row():
                txt = gr.Textbox(placeholder="Enter your travel query...", show_label=False, scale=8)
                send_btn = gr.Button("Send", scale=1)
                reset_btn = gr.Button("Reset", scale=1)
            txt.submit(self.respond, [txt, chatbot], [txt, chatbot])
            send_btn.click(self.respond, [txt, chatbot], [txt, chatbot])
            def do_reset():
                self.context = {}
                return [], []
            reset_btn.click(lambda: ("", []), outputs=[txt, chatbot]).then(lambda: do_reset(), outputs=[chatbot, chatbot])
        demo.launch()
