from agents.trip_planner_agent import get_trip_planner_agent
import gradio as gr


class DummyAgent:
    def plan_trip(self, prompt):
        return f"Echo: {prompt}"


class TripPlannerUI:
    def __init__(self):
        # self.agent = get_trip_planner_agent()
        self.agent = DummyAgent()

    def agent_chat(self, prompt):
        response = self.agent.plan_trip(prompt)

        return response

    def respond(self, message, chat_history):
        response = self.agent_chat(message)
        chat_history = chat_history or []
        chat_history.append((message, response))

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
            gr.Markdown(
                """
                # 🌍 Trip Planner AI
                Plan your next adventure with an AI-powered travel assistant. Ask about destinations, flights, activities, budgets, and more!
                """
            )
            chatbot = gr.Chatbot(show_label=False)

            with gr.Row():
                txt = gr.Textbox(placeholder="Enter your travel query...", show_label=False, scale=8)
                send_btn = gr.Button("Send", scale=0.5)

            txt.submit(self.respond, [txt, chatbot], [txt, chatbot])
            send_btn.click(self.respond, [txt, chatbot], [txt, chatbot])

        demo.launch()
