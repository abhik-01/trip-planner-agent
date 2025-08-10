"""
Centralized Prompt Management System for Trip Planner AI
Uses LangChain PromptTemplate for better performance and maintainability
"""

from langchain.prompts import PromptTemplate
from enum import Enum
from typing import Dict, List


class PromptType(Enum):
    """Enum for all prompt types in the system"""
    # Conversation Agent Prompts
    INTENT_CLASSIFICATION = "intent_classification"
    EXPLORATION_INTRO = "exploration_intro"
    EXPLORATION_GENERAL = "exploration_general"
    GENERAL_CHAT = "general_chat"
    PLANNING_DETAILS_EXTRACTION = "planning_details_extraction"
    DESTINATION_INFERENCE = "destination_inference"
    
    # Trip Inquiry Prompts
    WEATHER_INQUIRY = "weather_inquiry"
    ACTIVITY_INQUIRY = "activity_inquiry"
    NEARBY_INQUIRY = "nearby_inquiry"
    BUDGET_INQUIRY = "budget_inquiry"
    FLIGHT_INQUIRY = "flight_inquiry"
    GENERAL_TRIP_INQUIRY = "general_trip_inquiry"
    
    # Trip Planner Agent Prompts
    DURATION_SUGGESTION = "duration_suggestion"
    BEST_TIME_SUGGESTION = "best_time_suggestion"
    
    # Safety Prompts
    SAFETY_INPUT_SCREENING = "safety_input_screening"
    SAFETY_RESPONSE_VALIDATION = "safety_response_validation"
    SAFETY_DESTINATION_ASSESSMENT = "safety_destination_assessment"
    
    # Semantic Intelligence Prompts
    SEMANTIC_INTENT_FALLBACK = "semantic_intent_fallback"
    SEMANTIC_QUERY_CLASSIFICATION = "semantic_query_classification"
    SEMANTIC_CONTEXT_EXTRACTION = "semantic_context_extraction"
    SEMANTIC_FOLLOWUP_DETECTION = "semantic_followup_detection"
    FALLBACK_DETAIL_EXTRACTION = "fallback_detail_extraction"
    
    # Tool Prompts
    FLIGHT_LOCATION_RESOLUTION = "flight_location_resolution"
    FLIGHT_ERROR_MESSAGE = "flight_error_message"
    DESTINATION_SUGGESTION = "destination_suggestion"
    ACTIVITY_SUGGESTION = "activity_suggestion"
    BUDGET_ESTIMATION = "budget_estimation"
    ITINERARY_ASSEMBLY = "itinerary_assembly"


class PromptRegistry:
    """Centralized registry for all prompts with caching for performance"""
    
    _templates: Dict[PromptType, PromptTemplate] = {}
    _compiled = False
    
    @classmethod
    def _compile_templates(cls):
        """Compile all templates once for performance"""
        if cls._compiled:
            return
            
        # Conversation Agent Prompts
        cls._templates[PromptType.INTENT_CLASSIFICATION] = PromptTemplate(
            template="""You are a travel planning assistant. Analyze the user's intent and classify their request:

Recent conversation:
{recent_context}

Current user message: "{user_input}"

Classification rules:
- "explore": User wants travel destination suggestions, browsing travel options, or asking "where should I go"
- "plan": User has a specific destination in mind and wants to plan/book a trip (e.g., "plan a trip to Paris", "let's go to Japan")
- "chat": Everything else - general questions, greetings, non-travel topics, asking about agent capabilities, etc.

Important: Questions like "what can you do", "help me", "can you do anything else" are CHAT, not explore.
Only use "explore" when they specifically want travel destination suggestions.

Return JSON with: {{"intent": "explore|plan|chat", "exploring": "string or null", "planning_destination": "string or null", "ready_to_plan": true/false}}""",
            input_variables=["recent_context", "user_input"]
        )
        
        cls._templates[PromptType.EXPLORATION_INTRO] = PromptTemplate(
            template="""Based on the user's request "{user_input}", create a brief, enthusiastic introduction that sets up destination suggestions.

IMPORTANT: Do NOT suggest any specific destinations. Just create an engaging intro that leads into a list of suggestions.

Examples:
- "I'd love to help you discover amazing destinations! Here are some fantastic options:"
- "Great question! I have some exciting suggestions for you:"
- "Perfect timing for travel planning! Here are some wonderful places to consider:"

Keep it to 1 sentence maximum.""",
            input_variables=["user_input"]
        )
        
        cls._templates[PromptType.EXPLORATION_GENERAL] = PromptTemplate(
            template="""The user is exploring travel options: "{user_input}"

Provide helpful, enthusiastic suggestions. Be conversational and engaging. End with a question to keep the conversation flowing.""",
            input_variables=["user_input"]
        )
        
        cls._templates[PromptType.GENERAL_CHAT] = PromptTemplate(
            template="""You are a friendly travel planning assistant. The user said: "{user_input}"

Your role is to:
1. Respond naturally and helpfully to whatever they asked
2. If it's travel-related, be enthusiastic and helpful
3. If it's NOT travel-related (like asking about smartphones, weather in general, your capabilities, etc.):
   - Acknowledge their question politely
   - Give a brief, helpful response if appropriate
   - Gently redirect to travel planning without being pushy
   - Don't be aggressive about it

Examples:
- If they ask "what can you do?": Explain you're a travel planning assistant and what you can help with
- If they ask about non-travel topics: Politely acknowledge and naturally mention you specialize in travel
- If they're just chatting: Be friendly and eventually mention you're great at trip planning

Keep responses conversational, warm, and natural. Don't sound robotic or overly sales-focused.""",
            input_variables=["user_input"]
        )
        
        cls._templates[PromptType.PLANNING_DETAILS_EXTRACTION] = PromptTemplate(
            template="""Extract specific trip planning details from: "{user_input}"

Context from conversation: {chat_context}

Only extract details that are EXPLICITLY mentioned or clearly implied. 

Return JSON with any of these keys (only if stated/implied):
- destination: specific place name (if user says "plan a trip from delhi" and we were discussing Rajasthan, destination could be inferred)
- origin: departure city  
- date: travel date (YYYY-MM-DD if possible)
- duration: number of days
- travelers: number of people
- budget: budget amount

Current context: {context}
Return JSON only.""",
            input_variables=["user_input", "chat_context", "context"]
        )
        
        cls._templates[PromptType.DESTINATION_INFERENCE] = PromptTemplate(
            template="""The user said: "{user_input}"

Recently mentioned places in our conversation: {mentions}

Are they likely referring to one of these places for their trip planning? If yes, which one? If unclear, return "unclear".

Return just the place name or "unclear".""",
            input_variables=["user_input", "mentions"]
        )
        
        # Trip Inquiry Prompts
        cls._templates[PromptType.WEATHER_INQUIRY] = PromptTemplate(
            template="""The user asked: "{user_input}"

I have weather information for their {destination} trip:
Travel date: {start_date}
Weather data: {weather_data}

Provide a natural, helpful response about the weather. Be conversational and include practical travel advice based on the weather conditions. Keep it concise but informative.""",
            input_variables=["user_input", "destination", "start_date", "weather_data"]
        )
        
        cls._templates[PromptType.ACTIVITY_INQUIRY] = PromptTemplate(
            template="""The user asked: "{user_input}"

I have activity suggestions for their {destination} trip:
{activities_data}

Provide a natural, enthusiastic response about these activities. Help them understand what makes each activity special and offer to provide more details about any specific activity they're interested in. Be conversational and helpful.""",
            input_variables=["user_input", "destination", "activities_data"]
        )
        
        cls._templates[PromptType.NEARBY_INQUIRY] = PromptTemplate(
            template="""The user asked: "{user_input}"

I have information about nearby places around {destination}:
{nearby_data}

Provide a natural, informative response about these nearby places. Help them understand what's special about each location and how they might fit into their travel itinerary. Be enthusiastic and offer additional help.""",
            input_variables=["user_input", "destination", "nearby_data"]
        )
        
        cls._templates[PromptType.BUDGET_INQUIRY] = PromptTemplate(
            template="""The user asked: "{user_input}"

I have budget information for their {destination} trip:
Duration: {duration} days
Travelers: {travelers}
Budget breakdown: {budget_data}

Provide a natural, helpful response about the trip budget. Explain the costs in a conversational way and offer suggestions for saving money or adjusting the budget if needed. Be practical and supportive.""",
            input_variables=["user_input", "destination", "duration", "travelers", "budget_data"]
        )
        
        cls._templates[PromptType.FLIGHT_INQUIRY] = PromptTemplate(
            template="""The user asked: "{user_input}"

I have flight information for their trip:
{flight_summary}

Provide a natural, helpful response about the flight options. Be conversational and highlight the key details like prices, airlines, and timing. Offer to provide more specific details if they're interested in any particular flight. Be enthusiastic but practical.""",
            input_variables=["user_input", "flight_summary"]
        )
        
        cls._templates[PromptType.GENERAL_TRIP_INQUIRY] = PromptTemplate(
            template="""The user asked: "{user_input}"

Here's their trip information:
{trip_summary}

Provide a natural, enthusiastic summary of their trip plan. Highlight the exciting aspects and offer to provide more details about any specific aspect they're interested in. Be conversational and helpful.""",
            input_variables=["user_input", "trip_summary"]
        )
        
        # Trip Planner Agent Prompts
        cls._templates[PromptType.DURATION_SUGGESTION] = PromptTemplate(
            template="""Suggest an ideal trip duration for {destination} in 3-7 words. Consider typical tourist activities and travel distance. Just state the recommendation naturally, like "I'd recommend 5 days" or "A week would be perfect".""",
            input_variables=["destination"]
        )
        
        cls._templates[PromptType.BEST_TIME_SUGGESTION] = PromptTemplate(
            template="""What is the best time of year to visit {destination}? Consider weather, crowds, and local events. Suggest the best months and explain why in 2-3 sentences.""",
            input_variables=["destination"]
        )
        
        # Safety Prompts
        cls._templates[PromptType.SAFETY_INPUT_SCREENING] = PromptTemplate(
            template="""You are a safety advisor for a travel planning assistant. Analyze this user request for any safety concerns:

User request: "{user_input}"

Assess for:
1. Illegal activities (drug tourism, human trafficking, illegal border crossings, smuggling)
2. Dangerous/unethical requests (war tourism for wrong reasons, seeking conflict zones for harmful purposes)
3. Off-topic harmful content (violence, weapons, illegal advice unrelated to travel)
4. Irresponsible tourism (environmental destruction, cultural exploitation, animal abuse)
5. Inappropriate personal requests (dating, relationships, adult content)

IMPORTANT: Legitimate travel questions about challenging destinations are SAFE if the intent is proper tourism.
Examples of SAFE requests: "Is it safe to visit Syria?", "Travel advice for Afghanistan", "Precautions for visiting high-crime areas"
Examples of UNSAFE requests: "How to smuggle drugs", "Where to buy weapons", "How to exploit local communities"

Return JSON only:
{{
  "is_safe": true/false,
  "concern_type": "illegal|dangerous|harmful|off_topic|exploitation|inappropriate|safe",
  "explanation": "brief explanation of concern or why it's safe",
  "suggested_response": "polite redirect if unsafe, empty if safe"
}}

Focus on INTENT and CONTEXT - be permissive for legitimate travel planning.""",
            input_variables=["user_input"]
        )
        
        cls._templates[PromptType.SAFETY_RESPONSE_VALIDATION] = PromptTemplate(
            template="""Review this travel assistant response for safety and responsibility:

User Context: "{user_context}"
Agent Response: "{response}"

Check for:
1. Unsafe travel advice (missing critical safety warnings for dangerous areas)
2. Irresponsible recommendations (environmentally harmful, culturally insensitive suggestions)
3. Missing important safety information for risky destinations
4. Inappropriate content that slipped through
5. Encouraging illegal or unethical activities
6. Lack of responsible tourism principles

IMPORTANT: The response should be educational and helpful while being responsible.
Safety warnings should be present for genuinely risky destinations.

Return JSON only:
{{
  "is_safe": true/false,
  "issues": ["list of specific issues found"],
  "severity": "low|medium|high",
  "improved_response": "safer version with proper warnings if needed, empty if original is fine"
}}""",
            input_variables=["user_context", "response"]
        )
        
        cls._templates[PromptType.SAFETY_DESTINATION_ASSESSMENT] = PromptTemplate(
            template="""Is "{destination}" a destination that requires SPECIAL safety warnings beyond normal travel precautions?

ONLY mark as sensitive if there are SIGNIFICANT concerns such as:
- Active conflict zones or war areas
- Severe political instability with current violence
- Extreme crime rates that pose serious danger to tourists
- Current natural disasters or extreme weather threats
- Severe health crises (epidemics, lack of medical care)
- Government travel bans or strong advisory warnings
- Areas completely restricted to civilians

NORMAL travel considerations are NOT sensitive:
- Standard city crime rates (pickpocketing, etc.)
- Normal cultural differences
- Standard visa requirements
- Seasonal weather patterns
- General tourist precautions

Examples:
- Paris, Tokyo, New York, Mumbai, London: NOT sensitive (normal travel destinations)
- Active war zones, areas under martial law, epidemic zones: SENSITIVE

Return JSON only:
{{
  "is_sensitive": true/false,
  "risk_level": "low|medium|high",
  "main_concerns": ["list of SIGNIFICANT safety concerns only"]
}}

Be conservative - only flag truly dangerous destinations.""",
            input_variables=["destination"]
        )
        
        # Semantic Intelligence Prompts
        cls._templates[PromptType.SEMANTIC_INTENT_FALLBACK] = PromptTemplate(
            template="""Classify this user message into one category:

User: "{user_input}"

Categories:
- "explore": Asking for destination suggestions, travel ideas, place recommendations
- "plan": Ready to plan a specific trip, has destination in mind, wants itinerary
- "chat": General conversation, questions about capabilities, greetings, other topics

Consider natural language variations and synonyms.

Respond with JSON: {{"intent": "explore|plan|chat", "confidence": 0.0-1.0}}""",
            input_variables=["user_input"]
        )
        
        cls._templates[PromptType.SEMANTIC_QUERY_CLASSIFICATION] = PromptTemplate(
            template="""Classify this travel-related question into the most relevant category:

User question: "{user_input}"
Context: {destination}

Categories:
- weather: Questions about climate, temperature, rain, seasons, what to wear
- activities: Things to do, attractions, sightseeing, entertainment, experiences
- nearby: Places around/near destination, vicinity, surrounding areas
- budget: Costs, expenses, money, pricing, affordability
- flights: Air travel, airlines, airports, booking flights
- accommodation: Hotels, stays, lodging, where to sleep
- food: Restaurants, cuisine, dining, local food, eating
- general: Overall trip information, mixed topics, or unclear

Consider natural language and synonyms.

Respond with just the category name.""",
            input_variables=["user_input", "destination"]
        )
        
        cls._templates[PromptType.SEMANTIC_CONTEXT_EXTRACTION] = PromptTemplate(
            template="""Analyze this conversation for travel context:
"{recent_conversation}"

Extract specific details and return JSON:
{{
    "recent_mentions": ["specific place names, countries, regions mentioned"],
    "geographic_context": ["countries/regions discussed like 'India', 'Europe', 'Asia'"],
    "temporal_context": ["time/season references like 'October', 'winter', 'spring'"],
    "geographic_constraints": ["explicit geographic limits like 'in India', 'places in Europe', etc."]
}}

IMPORTANT: Pay special attention to geographic constraints like "places in [country]" or "destinations in [region]".
Only include actual mentions. Return {{}} if none found.""",
            input_variables=["recent_conversation"]
        )
        
        cls._templates[PromptType.SEMANTIC_FOLLOWUP_DETECTION] = PromptTemplate(
            template="""Analyze if this is a follow-up request for more suggestions:
"{user_input}"

Return only "yes" or "no" based on whether the user is asking for:
- More options/suggestions
- Additional alternatives  
- Different choices
- Other recommendations""",
            input_variables=["user_input"]
        )
        
        cls._templates[PromptType.FALLBACK_DETAIL_EXTRACTION] = PromptTemplate(
            template="""Extract trip planning details from: "{user_input}"

Return a JSON object with any found details:
{{
    "destination": "destination name if mentioned",
    "origin": "departure city if mentioned", 
    "duration": "number of days if mentioned",
    "travelers": "number of people if mentioned",
    "date": "travel date if mentioned"
}}

Only include fields with actual values. Return {{}} if no details found.""",
            input_variables=["user_input"]
        )
        
        # Tool Prompts
        cls._templates[PromptType.FLIGHT_LOCATION_RESOLUTION] = PromptTemplate(
            template="""The user mentioned "{location}" as a travel location. I need to find the best major city with an airport for this location.

If it's a:
- Country/State: Return the most popular tourist city with major airport
- Region: Return the main city/capital  
- City: Return the same city if it has an airport, or nearest major city
- Unclear: Ask for clarification

Examples:
- "Rajasthan" → "Jaipur" (major tourist city)
- "California" → "Los Angeles" (major hub)
- "Europe" → "Too broad, please specify country"

Best city with airport:""",
            input_variables=["location"]
        )
        
        cls._templates[PromptType.FLIGHT_ERROR_MESSAGE] = PromptTemplate(
            template="""A user is trying to book flights but we couldn't find an airport for "{location}" as their {location_type}.

Generate a helpful error message that:
1. Acknowledges the issue politely
2. Suggests nearby cities with airports if possible
3. Asks them to specify a different city

Keep it under 2 sentences and helpful.""",
            input_variables=["location", "location_type"]
        )
        
        cls._templates[PromptType.DESTINATION_SUGGESTION] = PromptTemplate(
            template="""Based on the user's preferences and context, suggest 8-10 diverse travel destinations: {preferences}

CRITICAL: Analyze the input for geographic constraints FIRST:
- If input contains "Geographic focus: [country/region]", suggest destinations ONLY within that area
- If input mentions "places in [country]", "destinations in [region]", suggest ONLY within that geographic boundary
- If input says "in India", suggest ONLY Indian destinations
- If input says "in Europe", suggest ONLY European destinations

Context Analysis:
- If this appears to be a follow-up request (words like "more", "other", "additional", "few more"), focus on destinations that complement previously mentioned places
- If specific regions, seasons, or interests are mentioned, prioritize those
- RESPECT geographic constraints above all else

For each destination, include:
1. Why it matches the preferences/context
2. Best time to visit
3. One unique highlight

Format as a numbered list. Prioritize:
- GEOGRAPHIC CONSTRAINT COMPLIANCE (most important)
- Geographic relevance to any mentioned regions
- Seasonal appropriateness if timing is specified
- Less obvious but excellent matches alongside popular choices
- Cultural and interest-based alignment

IMPORTANT: Promote responsible and sustainable tourism. If suggesting any destinations with cultural sensitivities or environmental concerns, include brief respectful notes about responsible travel practices.""",
            input_variables=["preferences"]
        )
        
        cls._templates[PromptType.ACTIVITY_SUGGESTION] = PromptTemplate(
            template="""List 8-10 popular and diverse activities that travelers can enjoy in {destination}. 
Include a brief description for each activity and organize them by type (cultural, outdoor, food, etc.). 
Format as a numbered list for easy reading.

IMPORTANT: Promote responsible and ethical tourism. Prioritize activities that:
- Respect local culture and communities
- Support sustainable and eco-friendly practices
- Avoid exploitation of people, animals, or environment
- Encourage cultural exchange and understanding""",
            input_variables=["destination"]
        )
        
        cls._templates[PromptType.BUDGET_ESTIMATION] = PromptTemplate(
            template="""Estimate a realistic (avoid overestimation) trip budget in INR (Indian Rupees) for the following:
Destination: {destination}
{flight_line}
Number of nights: {nights}
Number of travelers: {travelers}
Activities: {activities}
Provide accommodation, activities, food, local transport, and a 10% miscellaneous buffer. If flight cost unknown, omit it. 
Return a category breakdown plus total in INR (₹). Use current Indian pricing for all estimates.

IMPORTANT: Consider responsible tourism practices in budget suggestions:
- Prioritize locally-owned accommodations and businesses
- Include fair wages for local guides and service providers
- Account for sustainable and ethical activity choices""",
            input_variables=["destination", "flight_line", "nights", "travelers", "activities"]
        )
        
        cls._templates[PromptType.ITINERARY_ASSEMBLY] = PromptTemplate(
            template="""Create a comprehensive, engaging travel itinerary for {destination} using the following data:

📊 Available Data: {data_summary}

🗂️ Trip Details:
{trip_data}

📋 Requirements:
- Create a day-by-day plan if duration is specified
- Include all available information (flights, weather, activities, budget)
- Make it engaging and user-friendly
- Highlight important details with emojis
- Include practical travel tips
- Structure with clear headings and sections
- All prices should be in INR (Indian Rupees)
- Be enthusiastic but informative

🌍 RESPONSIBLE TOURISM GUIDELINES:
- Emphasize respect for local cultures, customs, and communities
- Promote sustainable travel practices and environmental consciousness
- Suggest supporting local businesses and fair-wage services
- Include cultural sensitivity tips where relevant
- Encourage meaningful cultural exchange and understanding""",
            input_variables=["destination", "data_summary", "trip_data"]
        )
        
        cls._compiled = True
    
    @classmethod
    def get_prompt(cls, prompt_type: PromptType) -> PromptTemplate:
        """Get a compiled prompt template by type"""
        cls._compile_templates()
        return cls._templates[prompt_type]
    
    @classmethod
    def format_prompt(cls, prompt_type: PromptType, **kwargs) -> str:
        """Format a prompt without caching for complex objects"""
        template = cls.get_prompt(prompt_type)
        return template.format(**kwargs)
    
    @classmethod
    def get_input_variables(cls, prompt_type: PromptType) -> List[str]:
        """Get the required input variables for a prompt"""
        template = cls.get_prompt(prompt_type)
        return template.input_variables


# Convenience functions for easy migration
def get_prompt(prompt_type: PromptType) -> PromptTemplate:
    """Get a prompt template"""
    return PromptRegistry.get_prompt(prompt_type)


def format_prompt(prompt_type: PromptType, **kwargs) -> str:
    """Format a prompt with the given variables"""
    return PromptRegistry.format_prompt(prompt_type, **kwargs)


# Pre-compile templates on import for better performance
PromptRegistry._compile_templates()
