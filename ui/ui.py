from agents.conversation_agent import run_conversation_graph
from asyncio import get_event_loop, new_event_loop, set_event_loop
from atexit import register
from concurrent.futures import ThreadPoolExecutor
from os import path as os_path
from time import sleep
from uuid import uuid4
from .templates import HTMLTemplates
import gradio as gr



class TripPlannerUI:
    def __init__(self):
        self.run_conversation_graph = run_conversation_graph
        self.executor = ThreadPoolExecutor(max_workers=3)
        # Ensure executor cleans up on process exit
        register(lambda: self.executor.shutdown(wait=False))

        # Load CSS from external file
        self.custom_css = self._load_css()

    def _load_css(self):
        """Load CSS from external file"""
        css_path = os_path.join(os_path.dirname(__file__), 'styles.css')
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return ""

    async def agent_chat_async(self, prompt, chat_history, context):
        """Async wrapper for agent chat to prevent UI freezing"""
        loop = get_event_loop()
        state = await loop.run_in_executor(
            self.executor, 
            self.run_conversation_graph, 
            prompt, chat_history, context
        )
        return state

    def agent_chat(self, prompt, chat_history, context):
        """Sync wrapper for async agent chat"""
        try:
            # Run async function in new event loop
            loop = new_event_loop()
            set_event_loop(loop)
            state = loop.run_until_complete(self.agent_chat_async(prompt, chat_history, context))
            loop.close()
            return state
        except Exception as e:
            return {"response": f"Error processing request: {str(e)}", "missing_info": False}

    def add_user_message(self, message, history, context_state, started_state, thread_id_state, welcome_area):
        """Immediately add user message to chat - no AI processing"""
        if not message.strip():
            return "", history, context_state, started_state, thread_id_state, gr.update()
        
        # Initialize per-session thread id and context
        context = dict(context_state or {})
        thread_id = thread_id_state or context.get('_thread_id')
        if not thread_id:
            thread_id = str(uuid4())
            context['_thread_id'] = thread_id
        
        # Handle first interaction - smooth fade then remove welcome content
        if not started_state:
            started_state = True
            welcome_area_update = gr.update(
                value=HTMLTemplates.get_welcome_fadeout_with_script(),
                visible=True
            )
        else:
            welcome_area_update = gr.update()
        
        # Add user message immediately to chat
        history.append({"role": "user", "content": message})
        
        # Add thinking indicator immediately
        thinking_html = HTMLTemplates.get_thinking_indicator()
        history.append({"role": "assistant", "content": thinking_html})
        
        return "", history, context, started_state, thread_id, welcome_area_update

    def process_ai_response(self, message, history, context_state, started_state, thread_id_state, welcome_area):
        """Process AI response with streaming - called after user message is added"""
        if not message.strip():
            yield "", history, context_state, started_state, thread_id_state, gr.update()
            return
            
        context = dict(context_state or {})
        thread_id = thread_id_state or context.get('_thread_id')
        
        # Prepare a clean recent chat history for the agent (exclude placeholders)
        def is_real_message(msg: dict) -> bool:
            if not isinstance(msg, dict):
                return False
            role = msg.get('role')
            content = str(msg.get('content', ''))
            if 'thinking-indicator' in content:
                return False
            return role in ('user', 'assistant') and bool(content.strip())

        recent_history = [m for m in history if is_real_message(m)][-8:]

        # Get AI response (this is the blocking operation, but user message is already shown)
        state = self.agent_chat(message, recent_history, context)
        if isinstance(state, dict):
            response = state.get('response', 'Sorry, something went wrong.')
            # Update session context if agent returned it
            if state.get('context'):
                context = state['context']
                # Ensure thread id persists
                context['_thread_id'] = thread_id
        else:
            response = str(state)
        
        # Stream the response word-by-word for a natural typing effect
        words = response.split()
        partial_words = []
        for w in words:
            partial_words.append(w)
            history[-1]["content"] = " ".join(partial_words).strip()
            yield "", history, context, started_state, thread_id, gr.update()
            sleep(0.03)
        
        # Ensure final response is complete
        history[-1]["content"] = response
        yield "", history, context, started_state, thread_id, gr.update()

    def _create_interface(self):
        """Create the Gradio interface with clean separation of concerns"""
        with gr.Blocks(title="Trip Planner AI", css=self.custom_css) as demo:
            # Session state
            ctx_state = gr.State({})
            started_state = gr.State(False)
            thread_state = gr.State("")
            
            # Welcome area that will be centered and fade out after first message
            welcome_area = gr.HTML(
                value=HTMLTemplates.get_welcome_area(),
                elem_id="welcome-area",
                visible=True
            )
            
            chatbot = gr.Chatbot(
                label="",
                type="messages",
                show_label=False,
                show_copy_button=False,
                elem_id="main-chatbot",
                visible=True,  # Always visible, controlled by CSS
            )
            
            # Modern ChatGPT-like input interface
            with gr.Row(elem_classes=["message-row"]):
                with gr.Column(elem_classes=["message-input-container"]):
                    with gr.Group(elem_classes=["message-input-wrapper"]):
                        txt = gr.Textbox(
                            placeholder="Message Trip Planner AI...",
                            show_label=False,
                            container=False,
                            elem_classes=["message-input"]
                        )
                        send_btn = gr.Button("↗", elem_classes=["send-button"], size="sm")
            
            # Store the current message for AI processing
            current_message_state = gr.State("")
            
            # Event handlers with immediate user message display
            def handle_submit(message, history, context_state, started_state, thread_id_state, welcome_area, current_msg_state):
                # Step 1: Immediately add user message (no blocking operations)
                empty_input, updated_history, updated_context, updated_started, updated_thread_id, updated_welcome = self.add_user_message(
                    message, history, context_state, started_state, thread_id_state, welcome_area
                )
                
                # Store the message for AI processing
                current_msg_state = message
                
                # Return the immediate update
                return empty_input, updated_history, updated_context, updated_started, updated_thread_id, updated_welcome, current_msg_state
            
            def handle_ai_response(message, history, context_state, started_state, thread_id_state, welcome_area, current_msg_state):
                # Step 2: Process AI response with streaming using the stored message
                yield from self.process_ai_response(current_msg_state, history, context_state, started_state, thread_id_state, welcome_area)
            
            # First event: immediately add user message
            txt_submit = txt.submit(
                handle_submit,
                [txt, chatbot, ctx_state, started_state, thread_state, welcome_area, current_message_state],
                [txt, chatbot, ctx_state, started_state, thread_state, welcome_area, current_message_state],
                queue=False  # No queue for immediate response
            )
            
            send_submit = send_btn.click(
                handle_submit,
                [txt, chatbot, ctx_state, started_state, thread_state, welcome_area, current_message_state],
                [txt, chatbot, ctx_state, started_state, thread_state, welcome_area, current_message_state],
                queue=False  # No queue for immediate response
            )
            
            # Second event: process AI response (chained after user message is added)
            txt_submit.then(
                handle_ai_response,
                [txt, chatbot, ctx_state, started_state, thread_state, welcome_area, current_message_state],
                [txt, chatbot, ctx_state, started_state, thread_state, welcome_area],
                queue=True
            )
            
            send_submit.then(
                handle_ai_response,
                [txt, chatbot, ctx_state, started_state, thread_state, welcome_area, current_message_state],
                [txt, chatbot, ctx_state, started_state, thread_state, welcome_area],
                queue=True
            )
        
        return demo

    def launch(self):
        """Launch the Trip Planner UI"""
        demo = self._create_interface()
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            show_api=False
        )


if __name__ == "__main__":
    ui = TripPlannerUI()
    ui.launch()
