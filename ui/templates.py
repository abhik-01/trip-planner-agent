"""
HTML Templates for Trip Planner UI
Contains all HTML content used in the interface
"""

class HTMLTemplates:
    
    @staticmethod
    def get_thinking_indicator():
        """HTML for the AI thinking indicator"""
        return '''<div class="thinking-indicator">
            <span>Trip Planner AI is thinking</span>
            <div class="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>'''
    
    @staticmethod
    def get_welcome_area():
        """HTML for the welcome area shown on page load"""
        return """
        <div class="welcome-container">
            <div class="welcome-content">
                <h1 class="welcome-title">🌍 Trip Planner AI</h1>
                <p class="welcome-subtitle">Welcome! I'm your AI travel assistant. Ask me about destinations, flights, weather, activities, or budget planning.</p>
            </div>
        </div>
        """
    
    @staticmethod
    def get_welcome_fadeout_with_script():
        """HTML for the welcome area with fade-out script"""
        return """
        <div class="welcome-container fade-out">
            <h1 class="welcome-title">🌍 Trip Planner AI</h1>
            <p class="welcome-subtitle">Your intelligent travel companion for unforgettable journeys</p>
        </div>
        <script>
        // Immediately remove the welcome content without any delay
        (function() {
            const welcomeArea = document.getElementById('welcome-area');
            if (welcomeArea) {
                welcomeArea.classList.add('hidden');
                welcomeArea.style.display = 'none';
                welcomeArea.style.height = '0px';
                welcomeArea.style.minHeight = '0px';
                welcomeArea.style.padding = '0px';
                welcomeArea.style.margin = '0px';
                welcomeArea.innerHTML = '';
                
                // Also hide parent containers that might be taking space
                let parent = welcomeArea.parentElement;
                while (parent && parent !== document.body) {
                    if (parent.children.length === 1) {
                        parent.style.display = 'none';
                        parent.style.height = '0px';
                        parent.style.minHeight = '0px';
                        parent.style.padding = '0px';
                        parent.style.margin = '0px';
                    }
                    parent = parent.parentElement;
                }
            }
        })();
        </script>
        """
