from agents.conversation_agent import run_conversation_graph
import gradio as gr
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time


class TripPlannerUI:
    def __init__(self):
        self.run_conversation_graph = run_conversation_graph
        self.context = {}
        self.executor = ThreadPoolExecutor(max_workers=3)

    async def agent_chat_async(self, prompt, chat_history):
        """Async wrapper for agent chat to prevent UI freezing"""
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(
            self.executor, 
            self.run_conversation_graph, 
            prompt, chat_history, self.context
        )
        if isinstance(state, dict) and 'context' in state and state['context']:
            self.context = state['context']
        return state

    def agent_chat(self, prompt, chat_history):
        """Sync wrapper for async agent chat"""
        try:
            # Run async function in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            state = loop.run_until_complete(self.agent_chat_async(prompt, chat_history))
            loop.close()
            return state
        except Exception as e:
            return {"response": f"Error processing request: {str(e)}", "missing_info": False}

    def process_message(self, message, history):
        """Process user message and get AI response with streaming"""
        if not message.strip():
            return "", history
        
        # Add user message immediately to chat
        history.append({"role": "user", "content": message})
        
        # Clear the input box immediately and update chat with user message
        yield "", history
        
        # Add assistant message with animated processing indicator immediately
        thinking_html = '''<div class="thinking-indicator">
            <span>Trip Planner AI is thinking</span>
            <div class="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>'''
        history.append({"role": "assistant", "content": thinking_html})
        yield "", history
        
        # Small delay to ensure the thinking indicator is shown
        time.sleep(0.1)
        
        # Get AI response
        state = self.agent_chat(message, [])
        if isinstance(state, dict):
            response = state.get('response', 'Sorry, something went wrong.')
        else:
            response = str(state)
        
        # Stream the response word by word (restored streaming)
        words = response.split()
        for i, word in enumerate(words):
            # Build partial response
            partial_response = " ".join(words[:i+1])
            history[-1]["content"] = partial_response
            
            # Yield the updated history for streaming effect
            yield "", history
            
            # Small delay for natural typing effect
            time.sleep(0.03)
        
        # Ensure final response is complete
        history[-1]["content"] = response
        return "", history

    def launch(self):
        # Custom CSS for modern ChatGPT-like interface
        custom_css = """
        .footer {display: none !important;}
        .gradio-container .footer {display: none !important;}
        footer {display: none !important;}
        
        /* Hide Gradio's default processing indicators */
        .loading,
        .loading-indicator,
        .progress-bar,
        .progress-text,
        .gradio-loading,
        .gradio-spinner,
        .spinner,
        .dots-loading,
        .dots-flashing,
        .generating,
        .processing,
        .gradio-chatbot .loading,
        .gradio-chatbot .generating,
        .gradio-chatbot .pending,
        .chatbot .pending,
        .pending {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }
        
        /* Hide any default "..." or processing text more aggressively */
        .chatbot .message:contains("..."),
        .gradio-chatbot .message:contains("..."),
        .gradio-chatbot .bot:contains("..."),
        .chatbot .bot:contains("...") {
            display: none !important;
        }
        
        /* Hide the actual gradio processing elements */
        .gradio-chatbot .wrap.generating,
        .gradio-chatbot .message.generating,
        .gradio-chatbot .pending,
        .gradio-chatbot .loading {
            display: none !important;
        }
        
        /* Prevent any flashing or brief appearances */
        .gradio-chatbot * {
            transition: none !important;
        }
        
        /* Hide chatbot clear/delete buttons */
        .chatbot .clear-button,
        .chatbot .copy-button,
        .gradio-chatbot .clear-button,
        .gradio-chatbot .copy-button,
        button[title="Clear"],
        button[aria-label="Clear"],
        .chatbot-header button,
        .chatbot-controls button,
        .chatbot button[aria-label*="clear"],
        .chatbot button[aria-label*="Clear"],
        .chatbot button[aria-label*="delete"],
        .chatbot button[aria-label*="Delete"],
        .gradio-chatbot button[aria-label*="clear"],
        .gradio-chatbot button[aria-label*="Clear"],
        .gradio-chatbot button[aria-label*="delete"],
        .gradio-chatbot button[aria-label*="Delete"],
        .gradio-chatbot .icon-button,
        .chatbot .icon-button {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Animated thinking indicator */
        .thinking-indicator {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            color: #8e8ea0;
            font-style: italic;
        }
        
        .thinking-dots {
            display: inline-flex;
            gap: 2px;
        }
        
        .thinking-dots span {
            width: 4px;
            height: 4px;
            background-color: #8e8ea0;
            border-radius: 50%;
            animation: thinking 1.4s ease-in-out infinite both;
        }
        
        .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
        .thinking-dots span:nth-child(3) { animation-delay: 0s; }
        
        @keyframes thinking {
            0%, 80%, 100% {
                opacity: 0.3;
                transform: scale(0.8);
            }
            40% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        /* Hide the default message label */
        .input-container label {
            display: none !important;
        }
        
        /* Target the specific chatbot and make it completely seamless */
        #main-chatbot,
        #main-chatbot > div,
        #main-chatbot .gradio-chatbot,
        #main-chatbot .wrap,
        #main-chatbot .svelte-1eq475l,
        .gradio-chatbot,
        .chatbot {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            max-width: 900px !important;
            margin: 0 auto !important;
        }
        
        /* Remove all possible container styling */
        #main-chatbot *,
        .gradio-chatbot *,
        .chatbot * {
            background: transparent !important;
            background-color: transparent !important;
        }
        
        /* Specifically target common Gradio container classes */
        .svelte-1eq475l,
        .svelte-1qjjws6,
        .block,
        .gradio-block {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Remove any remaining borders or backgrounds */
        [data-testid="chatbot"],
        [class*="chatbot"],
        [class*="gradio-chatbot"] {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Remove borders from all messages */
        #main-chatbot .message,
        #main-chatbot .user,
        #main-chatbot .bot,
        #main-chatbot .assistant,
        .gradio-chatbot .message,
        .gradio-chatbot .user,
        .gradio-chatbot .bot,
        .gradio-chatbot .assistant,
        .chatbot .message,
        .chatbot .user,
        .chatbot .bot,
        .chatbot .assistant {
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Style user messages with distinct background - no size changes */
        #main-chatbot .user,
        #main-chatbot .user > div,
        .gradio-chatbot .user,
        .gradio-chatbot .user > div,
        .chatbot .user,
        .chatbot .user > div,
        .message[data-role="user"],
        .message.user {
            background: #2f2f2f !important;
            background-color: #2f2f2f !important;
            border-radius: 12px !important;
            border: none !important;
            box-shadow: none !important;
            opacity: 1 !important;
            transform: translateY(0) translateX(0) !important;
        }
        
        /* Style agent messages with transparent background - no size changes */
        #main-chatbot .bot,
        #main-chatbot .bot > div,
        #main-chatbot .assistant,
        #main-chatbot .assistant > div,
        .gradio-chatbot .bot,
        .gradio-chatbot .bot > div,
        .gradio-chatbot .assistant,
        .gradio-chatbot .assistant > div,
        .chatbot .bot,
        .chatbot .bot > div,
        .chatbot .assistant,
        .chatbot .assistant > div,
        .message[data-role="assistant"],
        .message.assistant,
        .message.bot {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            opacity: 1 !important;
            transform: translateY(0) translateX(0) !important;
        }
        
        /* Smooth message appearance animations - only when specifically triggered */
        .message-animate-in-right {
            animation: slideInRight 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards !important;
        }
        
        .message-animate-in-left {
            animation: slideInLeft 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards !important;
        }
        
        /* Smooth message appearance animations */
        @keyframes slideInRight {
            0% {
                opacity: 0;
                transform: translateY(20px) translateX(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0) translateX(0);
            }
        }
        
        @keyframes slideInLeft {
            0% {
                opacity: 0;
                transform: translateY(20px) translateX(-20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0) translateX(0);
            }
        }
        
        /* Gentle pulse for thinking indicator */
        .thinking-indicator {
            opacity: 1 !important;
            transform: translateY(0) !important;
        }
        
        /* Hide scrollbars */
        #main-chatbot,
        #main-chatbot *,
        .gradio-chatbot,
        .gradio-chatbot *,
        .chatbot,
        .chatbot * {
            scrollbar-width: none !important; /* Firefox */
            -ms-overflow-style: none !important; /* IE and Edge */
        }
        
        /* Hide scrollbars for Webkit browsers */
        #main-chatbot::-webkit-scrollbar,
        #main-chatbot *::-webkit-scrollbar,
        .gradio-chatbot::-webkit-scrollbar,
        .gradio-chatbot *::-webkit-scrollbar,
        .chatbot::-webkit-scrollbar,
        .chatbot *::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        
        /* Modern ChatGPT-like input styling */
        .message-input-container {
            max-width: 100% !important;
            margin: 0 auto !important;
            padding: 20px !important;
            transform: translateY(0) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        /* Main container - this is the parent textbox that should get the styling */
        .message-input-wrapper {
            position: relative !important;
            width: 100% !important;
            max-width: 800px !important;
            margin: 0 auto !important;
            background: #2f2f2f !important;
            border: 1px solid #565869 !important;
            border-radius: 24px !important;
            padding: 12px 50px 12px 16px !important;
            box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            transform: translateY(0) scale(1) !important;
        }
        
        /* Focus state on the parent container */
        .message-input-wrapper:focus-within {
            border-color: #10a37f !important;
            box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2) !important;
            transform: translateY(-1px) scale(1.002) !important;
        }
        
        /* Processing state styling for parent container */
        .message-input-wrapper.pending {
            border-color: #ff6b35 !important;
            box-shadow: 0 0 0 2px rgba(255, 107, 53, 0.3) !important;
            transform: translateY(-1px) scale(1.001) !important;
        }
        
        /* Remove all styling from the inner textbox */
        .message-input-wrapper input, 
        .message-input-wrapper textarea {
            background: transparent !important;
            border: none !important;
            outline: none !important;
            color: #ececf1 !important;
            font-size: 16px !important;
            line-height: 24px !important;
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
            resize: none !important;
            min-height: 24px !important;
            max-height: 200px !important;
            box-shadow: none !important;
        }
        
        /* Remove any default gradio styling from inner textbox */
        .message-input-wrapper .gradio-textbox,
        .message-input-wrapper .gradio-textbox > label,
        .message-input-wrapper .gradio-textbox > label > div,
        .message-input-wrapper .gradio-textbox .wrap,
        .message-input-wrapper .gradio-textbox input,
        .message-input-wrapper .gradio-textbox textarea {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Force remove all borders and backgrounds from inner elements */
        .message-input-wrapper * {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        
        /* But keep the parent container styling */
        .message-input-wrapper {
            background: #2f2f2f !important;
            border: 1px solid #565869 !important;
            box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
        }
        
        .message-input-wrapper input::placeholder,
        .message-input-wrapper textarea::placeholder {
            color: #8e8ea0 !important;
        }
        
        /* Send button embedded in the container */
        .send-button {
            position: absolute !important;
            right: 8px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            background: #10a37f !important;
            color: white !important;
            border: none !important;
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            font-size: 14px !important;
            font-weight: bold !important;
            z-index: 10 !important;
            min-width: 32px !important;
            padding: 0 !important;
        }
        
        .send-button:hover {
            background: #0d8d6a !important;
            transform: translateY(-50%) scale(1.1) rotate(15deg) !important;
            box-shadow: 0 4px 12px rgba(16, 163, 127, 0.3) !important;
        }
        
        .send-button:active {
            transform: translateY(-50%) scale(0.95) !important;
            transition: all 0.1s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .send-button:disabled {
            background: #565869 !important;
            cursor: not-allowed !important;
            transform: translateY(-50%) scale(0.9) !important;
        }
        
        /* Hide default gradio row styling */
        .message-row .gap {
            gap: 0 !important;
        }
        
        .message-row {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Welcome area styling - ChatGPT like centered welcome */
        .welcome-container {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 60vh !important;
            text-align: center !important;
            padding: 40px 20px !important;
            opacity: 1 !important;
            transform: translateY(0) !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .welcome-content {
            max-width: 600px !important;
            margin: 0 auto !important;
            transform: translateY(0) !important;
            transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .welcome-title {
            font-size: 48px !important;
            font-weight: 600 !important;
            margin: 0 0 16px 0 !important;
            color: #ececf1 !important;
            background: transparent !important;
            transform: translateY(0) !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) 0.1s !important;
        }
        
        .welcome-subtitle {
            font-size: 20px !important;
            color: #8e8ea0 !important;
            margin: 0 !important;
            line-height: 1.5 !important;
            background: transparent !important;
            transform: translateY(0) !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) 0.2s !important;
        }
        
        /* Smooth fade-out animation for welcome area */
        .welcome-container.fade-out {
            opacity: 0 !important;
            transform: translateY(-20px) !important;
            pointer-events: none !important;
        }
        
        .welcome-container.fade-out .welcome-content {
            transform: translateY(-10px) !important;
        }
        
        .welcome-container.fade-out .welcome-title {
            transform: translateY(-15px) !important;
            opacity: 0 !important;
        }
        
        .welcome-container.fade-out .welcome-subtitle {
            transform: translateY(-10px) !important;
            opacity: 0 !important;
        }
        
        /* Hide welcome area after animation */
        .welcome-container.hidden {
            display: none !important;
        }
        
        /* Smooth chatbot appearance */
        #main-chatbot {
            opacity: 1 !important;
            transform: translateY(0) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            display: block !important;
        }
        
        #main-chatbot.welcome-active {
            opacity: 0 !important;
            transform: translateY(20px) !important;
            pointer-events: none !important;
            height: 0 !important;
            overflow: hidden !important;
        }
        
        #main-chatbot.chat-active {
            opacity: 1 !important;
            transform: translateY(0) !important;
            pointer-events: auto !important;
            height: auto !important;
            overflow: visible !important;
        }
        """
        
        # Clean Gradio interface with no custom styling
        with gr.Blocks(title="Trip Planner AI", css=custom_css) as demo:
            # Welcome area that will be centered and hidden after first message
            welcome_area = gr.HTML(
                value="""
                <div id="welcome-container" class="welcome-container">
                    <div class="welcome-content">
                        <h1 class="welcome-title">🌍 Trip Planner AI</h1>
                        <p class="welcome-subtitle">Welcome! I'm your AI travel assistant. Ask me about destinations, flights, weather, activities, or budget planning.</p>
                    </div>
                </div>
                """,
                elem_id="welcome-area"
            )
            
            chatbot = gr.Chatbot(
                label="",
                type="messages",
                show_label=False,
                show_copy_button=False,
                elem_id="main-chatbot",
                # height=400
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
            
            # Event handlers with streaming enabled
            txt.submit(
                self.process_message,
                [txt, chatbot],
                [txt, chatbot],
                queue=True
            )
            
            send_btn.click(
                self.process_message,
                [txt, chatbot],
                [txt, chatbot],
                queue=True
            )
            
            # JavaScript to hide welcome area when chat starts
            demo.load(None, None, None, js="""
            function() {
                function setupWelcomeTransition() {
                    const chatbot = document.querySelector('#main-chatbot');
                    const welcomeContainer = document.querySelector('.welcome-container');
                    let hasTransitioned = false;
                    
                    if (chatbot && welcomeContainer) {
                        // Function to check if chat has real messages
                        function hasRealMessages() {
                            const chatContent = chatbot.innerHTML;
                            return chatContent.includes('role="user"') || 
                                   chatContent.includes('role="assistant"') ||
                                   chatContent.includes('class="user"') ||
                                   chatContent.includes('class="bot"') ||
                                   (chatContent.length > 100 && !chatContent.includes('welcome'));
                        }
                        
                        // Observer to watch for message changes
                        const observer = new MutationObserver(function(mutations) {
                            const hasMessages = hasRealMessages();
                            
                            // Only transition once when first message appears
                            if (hasMessages && !hasTransitioned) {
                                hasTransitioned = true;
                                
                                // Smooth transition: welcome out, chat in
                                welcomeContainer.classList.add('fade-out');
                                chatbot.classList.remove('welcome-active');
                                chatbot.classList.add('chat-active');
                                
                                // Hide welcome after transition
                                setTimeout(() => {
                                    welcomeContainer.style.display = 'none';
                                    welcomeContainer.classList.add('hidden');
                                }, 400);
                                
                            } else if (!hasMessages && hasTransitioned) {
                                // Reset if chat is cleared
                                hasTransitioned = false;
                                welcomeContainer.style.display = 'flex';
                                welcomeContainer.classList.remove('hidden', 'fade-out');
                                chatbot.classList.remove('chat-active');
                                chatbot.classList.add('welcome-active');
                            }
                        });
                        
                        observer.observe(chatbot, {
                            childList: true,
                            subtree: true,
                            characterData: true
                        });
                        
                        // Initial state setup
                        if (hasRealMessages()) {
                            hasTransitioned = true;
                            welcomeContainer.style.display = 'none';
                            welcomeContainer.classList.add('hidden');
                            chatbot.classList.add('chat-active');
                        } else {
                            chatbot.classList.add('welcome-active');
                        }
                    }
                }
                
                // Run after DOM is ready
                setTimeout(setupWelcomeTransition, 300);
            }
            """)
        
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            show_api=False
        )


if __name__ == "__main__":
    ui = TripPlannerUI()
    ui.launch()
