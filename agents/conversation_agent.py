from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, List
import re
from datetime import date
from concurrent.futures import ThreadPoolExecutor
import time

from tools.destination import get_destination_tool
from classTypes.class_types import TripPlannerState
from agents.trip_planner_agent import trip_planner_node
from utils.set_llm import get_llm

ConversationAgentState = TripPlannerState

def _classify_user_intent(user_input: str, chat_history: List[Dict[str,str]]) -> Dict[str, Any]:
    """Use LLM to understand what the user actually wants"""
    llm = get_llm(temperature=0.3)
    
    # Build recent context
    recent_context = ""
    for msg in chat_history[-6:]:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            recent_context += f"{role}: {content}\n"
    
    prompt = f"""Analyze the user's intent and classify their request:

Recent conversation:
{recent_context}

Current user message: "{user_input}"

Determine:
1. Intent: "explore" (asking for suggestions, browsing options) OR "plan" (ready to book/plan specific trip) OR "chat" (general conversation)
2. If exploring, what are they exploring? (destinations, activities, etc.)
3. If planning, what specific destination/details do they have?
4. Are they asking for suggestions or ready to commit to planning?

Return JSON with: {{"intent": "explore|plan|chat", "exploring": "string or null", "planning_destination": "string or null", "ready_to_plan": true/false}}
"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        # Extract JSON from response
        import json
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Intent classification failed: {e}")
    
    # Fallback heuristics
    lower_input = user_input.lower()
    if any(word in lower_input for word in ['suggest', 'recommend', 'ideas', 'where', 'places', 'destinations']):
        return {"intent": "explore", "exploring": "destinations", "planning_destination": None, "ready_to_plan": False}
    elif any(word in lower_input for word in ['plan', 'book', 'itinerary', 'lets go', "let's go"]):
        return {"intent": "plan", "exploring": None, "planning_destination": None, "ready_to_plan": True}
    elif any(word in lower_input for word in [
        'flight', 'price', 'cost', 'fare', 'ticket', 'how much',  # flights
        'weather', 'temperature', 'rain', 'climate', 'forecast',  # weather
        'activities', 'things to do', 'attractions', 'sightseeing',  # activities
        'nearby', 'around', 'close to', 'near', 'vicinity',  # nearby places
        'budget', 'expense', 'money', 'spend'  # budget
    ]):
        # All trip-specific questions should be handled as chat to reference previous data
        return {"intent": "chat", "exploring": None, "planning_destination": None, "ready_to_plan": False}
    else:
        return {"intent": "chat", "exploring": None, "planning_destination": None, "ready_to_plan": False}

def _handle_exploration(user_input: str, intent_data: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Handle exploration requests - suggestions, browsing, etc. with async optimization"""
    
    exploring = intent_data.get('exploring', '').lower()
    
    if 'destination' in exploring or 'places' in exploring or not exploring:
        # Use destination suggestion tool with parallel LLM processing
        try:
            def get_suggestions():
                dest_tool = get_destination_tool()
                return dest_tool.func(user_input)
            
            def get_context_response():
                llm = get_llm()
                prompt = f"""Based on the user's request "{user_input}", create a brief, enthusiastic introduction for destination suggestions. Keep it to 1-2 sentences."""
                return llm.invoke(prompt).content.strip()
            
            # Run both operations in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                suggestions_future = executor.submit(get_suggestions)
                intro_future = executor.submit(get_context_response)
                
                # Get results with timeout
                try:
                    suggestions = suggestions_future.result(timeout=10)
                    intro = intro_future.result(timeout=5)
                except Exception:
                    # Fallback if parallel execution fails
                    suggestions = get_suggestions()
                    intro = "Here are some great destination ideas for you:"
            
            # Combine results
            response = f"{intro}\n\n{suggestions}\n\nWhich of these catches your interest? Just tell me the destination name when you're ready to start planning!"
            return response
            
        except Exception as e:
            print(f"[DEBUG] Exploration tool failed: {e}")
            return "I'd love to suggest some amazing destinations! Could you tell me what kind of experience you're looking for?"
    
    # For other types of exploration, use LLM
    llm = get_llm()
    prompt = f"""The user is exploring travel options: "{user_input}"

Provide helpful, enthusiastic suggestions. Be conversational and engaging. End with a question to keep the conversation flowing.
"""
    
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return "That sounds exciting! What kind of travel experience are you looking for?"

def _handle_general_chat(user_input: str, context: Dict[str, Any]) -> str:
    """Handle general conversation, greetings, and specific questions about previous plans"""
    
    # Check if user is asking about specific aspects of their previous trip planning
    if _is_trip_specific_inquiry(user_input, context):
        return _handle_trip_specific_inquiry(user_input, context)
    
    llm = get_llm()
    
    prompt = f"""You are a friendly travel planning assistant. The user said: "{user_input}"

Respond naturally and conversationally. If it's a greeting, greet back warmly. If they seem interested in travel, gently guide toward travel planning. Keep it brief and friendly.
"""
    
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return "Hello! I'm here to help you plan amazing trips. What kind of adventure are you thinking about?"


def _is_trip_specific_inquiry(user_input: str, context: Dict[str, Any]) -> bool:
    """Check if the user is asking about specific aspects of their previous trip"""
    
    # Must have previous trip data to answer specific questions
    if not context.get('last_trip_data') or not context.get('tool_results'):
        return False
    
    inquiry_keywords = {
        'weather': ['weather', 'temperature', 'rain', 'climate', 'forecast'],
        'activities': ['activities', 'things to do', 'attractions', 'places to visit', 'sightseeing'],
        'nearby': ['nearby', 'around', 'close to', 'near', 'vicinity', 'surroundings'],
        'budget': ['budget', 'cost', 'expense', 'price', 'money', 'spend'],
        'flights': ['flight', 'plane', 'airline', 'airport', 'fly'],
        'accommodation': ['hotel', 'stay', 'accommodation', 'lodging'],
        'food': ['food', 'restaurant', 'eat', 'cuisine', 'dining']
    }
    
    user_lower = user_input.lower()
    for category, keywords in inquiry_keywords.items():
        if any(keyword in user_lower for keyword in keywords):
            return True
    
    return False


def _handle_trip_specific_inquiry(user_input: str, context: Dict[str, Any]) -> str:
    """Handle specific questions about different aspects of the planned trip"""
    
    last_trip_data = context.get('last_trip_data', {})
    tool_results = context.get('tool_results', {})
    destination = last_trip_data.get('destination', 'your destination')
    
    user_lower = user_input.lower()
    
    # Weather inquiries
    if any(keyword in user_lower for keyword in ['weather', 'temperature', 'rain', 'climate', 'forecast']):
        return _handle_weather_inquiry(user_input, last_trip_data, tool_results)
    
    # Activity inquiries  
    elif any(keyword in user_lower for keyword in ['activities', 'things to do', 'attractions', 'places to visit', 'sightseeing']):
        return _handle_activity_inquiry(user_input, last_trip_data, tool_results)
    
    # Nearby places inquiries
    elif any(keyword in user_lower for keyword in ['nearby', 'around', 'close to', 'near', 'vicinity', 'surroundings']):
        return _handle_nearby_inquiry(user_input, last_trip_data, tool_results)
    
    # Budget inquiries
    elif any(keyword in user_lower for keyword in ['budget', 'cost', 'expense', 'money', 'spend']):
        return _handle_budget_inquiry(user_input, last_trip_data, tool_results)
    
    # Flight inquiries (existing function)
    elif any(keyword in user_lower for keyword in ['flight', 'plane', 'airline', 'airport', 'fly']):
        return _handle_flight_inquiry(user_input, context)
    
    # General trip summary
    else:
        return _handle_general_trip_inquiry(user_input, last_trip_data, tool_results)


def _handle_weather_inquiry(user_input: str, trip_data: dict, tool_results: dict) -> str:
    """Handle weather-related questions using LLM for natural responses"""
    
    destination = trip_data.get('destination')
    start_date = trip_data.get('start_date')
    weather_data = tool_results.get('weather')
    
    if not weather_data:
        return f"I don't have weather information for {destination} from our previous planning. Would you like me to check the current weather forecast?"
    
    if 'error' in str(weather_data).lower():
        return f"I had trouble getting weather data for {destination} earlier: {weather_data}. Would you like me to try again?"
    
    # Use LLM to generate natural response
    llm = get_llm()
    prompt = f"""The user asked: "{user_input}"

I have weather information for their {destination} trip:
Travel date: {start_date or 'Not specified'}
Weather data: {weather_data}

Provide a natural, helpful response about the weather. Be conversational and include practical travel advice based on the weather conditions. Keep it concise but informative.
"""
    
    try:
        response = llm.invoke(prompt)
        return str(getattr(response, 'content', response))
    except Exception:
        # Fallback to basic response
        return f"Here's the weather for your {destination} trip: {weather_data}"


def _handle_activity_inquiry(user_input: str, trip_data: dict, tool_results: dict) -> str:
    """Handle activity-related questions using LLM for natural responses"""
    
    destination = trip_data.get('destination')
    activities_data = tool_results.get('activities')
    
    if not activities_data:
        return f"I don't have activity suggestions for {destination} from our previous planning. Would you like me to suggest some activities for you?"
    
    # Use LLM to generate natural response
    llm = get_llm()
    prompt = f"""The user asked: "{user_input}"

I have activity suggestions for their {destination} trip:
{activities_data}

Provide a natural, enthusiastic response about these activities. Help them understand what makes each activity special and offer to provide more details about any specific activity they're interested in. Be conversational and helpful.
"""
    
    try:
        response = llm.invoke(prompt)
        return str(getattr(response, 'content', response))
    except Exception:
        # Fallback to basic response
        return f"Here are the activities for your {destination} trip: {activities_data}"


def _handle_nearby_inquiry(user_input: str, trip_data: dict, tool_results: dict) -> str:
    """Handle nearby places questions using LLM for natural responses"""
    
    destination = trip_data.get('destination')
    nearby_data = tool_results.get('nearby')
    
    if not nearby_data:
        return f"I don't have nearby places information for {destination} from our previous planning. Would you like me to find nearby attractions and points of interest?"
    
    if 'error' in str(nearby_data).lower() or 'not found' in str(nearby_data).lower():
        return f"I had trouble finding nearby places for {destination}: {nearby_data}. Would you like me to search again?"
    
    # Use LLM to generate natural response
    llm = get_llm()
    prompt = f"""The user asked: "{user_input}"

I have information about nearby places around {destination}:
{nearby_data}

Provide a natural, informative response about these nearby places. Help them understand what's special about each location and how they might fit into their travel itinerary. Be enthusiastic and offer additional help.
"""
    
    try:
        response = llm.invoke(prompt)
        return str(getattr(response, 'content', response))
    except Exception:
        # Fallback to basic response
        return f"Here are nearby places around {destination}: {nearby_data}"


def _handle_budget_inquiry(user_input: str, trip_data: dict, tool_results: dict) -> str:
    """Handle budget-related questions using LLM for natural responses"""
    
    destination = trip_data.get('destination')
    duration = trip_data.get('duration_days')
    travelers = trip_data.get('number_of_travelers')
    budget_data = tool_results.get('budget')
    
    if not budget_data:
        return f"I don't have budget information for your {destination} trip from our previous planning. Would you like me to create a budget estimate for you?"
    
    # Use LLM to generate natural response
    llm = get_llm()
    prompt = f"""The user asked: "{user_input}"

I have budget information for their {destination} trip:
Duration: {duration} days
Travelers: {travelers}
Budget breakdown: {budget_data}

Provide a natural, helpful response about the trip budget. Explain the costs in a conversational way and offer suggestions for saving money or adjusting the budget if needed. Be practical and supportive.
"""
    
    try:
        response = llm.invoke(prompt)
        return str(getattr(response, 'content', response))
    except Exception:
        # Fallback to basic response
        return f"Here's your {destination} trip budget: {budget_data}"


def _handle_general_trip_inquiry(user_input: str, trip_data: dict, tool_results: dict) -> str:
    """Handle general questions about the planned trip using LLM for natural responses"""
    
    destination = trip_data.get('destination')
    
    if not destination:
        return "I don't have information about a previous trip. Would you like to start planning a new trip?"
    
    # Use LLM to generate natural response about the trip
    llm = get_llm()
    
    # Prepare trip summary
    trip_summary = f"""
Destination: {destination}
Duration: {trip_data.get('duration_days', 'Not specified')} days
Travelers: {trip_data.get('number_of_travelers', 'Not specified')}
Start Date: {trip_data.get('start_date', 'Not specified')}
Departure City: {trip_data.get('user_city', 'Not specified')}

Available Information:
"""
    
    if tool_results.get('flights'):
        trip_summary += f"- Flight options: {len(tool_results['flights']) if isinstance(tool_results['flights'], list) else 'Available'}\n"
    if tool_results.get('weather'):
        trip_summary += "- Weather forecast: Available\n"
    if tool_results.get('activities'):
        trip_summary += "- Activity suggestions: Available\n"
    if tool_results.get('nearby'):
        trip_summary += "- Nearby places: Available\n"
    if tool_results.get('budget'):
        trip_summary += "- Budget breakdown: Available\n"
    
    prompt = f"""The user asked: "{user_input}"

Here's their trip information:
{trip_summary}

Provide a natural, enthusiastic summary of their trip plan. Highlight the exciting aspects and offer to provide more details about any specific aspect they're interested in. Be conversational and helpful.
"""
    
    try:
        response = llm.invoke(prompt)
        return str(getattr(response, 'content', response))
    except Exception:
        # Fallback to basic response
        available_info = []
        if tool_results.get('flights'):
            available_info.append("flight options")
        if tool_results.get('weather'):
            available_info.append("weather forecast")
        if tool_results.get('activities'):
            available_info.append("activity suggestions")
        if tool_results.get('nearby'):
            available_info.append("nearby places")
        if tool_results.get('budget'):
            available_info.append("budget breakdown")
        
        if available_info:
            return f"I have {', '.join(available_info)} for your {destination} trip. What would you like to know more about?"
        else:
            return f"I have basic information about your {destination} trip. Would you like me to help plan more details?"


def _handle_flight_inquiry(user_input: str, context: Dict[str, Any]) -> str:
    """Handle specific flight-related questions using LLM for natural responses"""
    
    # Check if we have previous trip data
    last_trip_data = context.get('last_trip_data')
    tool_results = context.get('tool_results')
    
    if not last_trip_data or not tool_results or 'flights' not in tool_results:
        # No flight data available, provide helpful guidance
        return ("I don't have flight information from our previous conversation. When I create a complete trip plan, I gather flight details including prices. Would you like me to search for flights for a specific route? Please provide your departure city, destination, and travel date.")
    
    flights_data = tool_results['flights']
    destination = last_trip_data.get('destination')
    user_city = last_trip_data.get('user_city')
    start_date = last_trip_data.get('start_date')
    
    # Check if there was an error in flight search
    if isinstance(flights_data, list) and flights_data and 'error' in flights_data[0]:
        error_msg = flights_data[0]['error']
        return f"I had trouble finding flights for your {destination} trip: {error_msg}. Would you like me to try searching again with different criteria?"
    
    if not isinstance(flights_data, list) or not flights_data:
        return f"I couldn't find flight options for your {destination} trip from {user_city}. This might be because {destination} doesn't have a direct airport or the route needs connecting flights."
    
    # Use LLM to generate natural response about flights
    llm = get_llm()
    
    # Prepare flight summary
    flight_summary = f"""
Route: {user_city} to {destination}
Travel date: {start_date or 'Not specified'}
Number of flight options: {len(flights_data)}

Flight details:
"""
    
    for i, flight in enumerate(flights_data[:3], 1):  # Show top 3 flights
        flight_summary += f"""
Option {i}: {flight.get('airline', 'Unknown')}
- Price: ₹{flight.get('price_in_inr', 'N/A')} INR (Original: {flight.get('currency', '')} {flight.get('price', 'N/A')})
- Departure: {flight.get('departure_time', 'N/A')}
- Route: {flight.get('departure_airport', 'N/A')} → {flight.get('arrival_airport', 'N/A')}
"""
    
    prompt = f"""The user asked: "{user_input}"

I have flight information for their trip:
{flight_summary}

Provide a natural, helpful response about the flight options. Be conversational and highlight the key details like prices, airlines, and timing. Offer to provide more specific details if they're interested in any particular flight. Be enthusiastic but practical.
"""
    
    try:
        response = llm.invoke(prompt)
        return str(getattr(response, 'content', response))
    except Exception:
        # Fallback to structured response
        cheapest_flight = min(flights_data, key=lambda x: x.get('price', float('inf')))
        return f"I found {len(flights_data)} flight options for your {destination} trip from {user_city}. The cheapest option is {cheapest_flight.get('airline', 'N/A')} at ₹{cheapest_flight.get('price_in_inr', 'N/A')} INR. Would you like more details about the flight options?"

def _extract_planning_details(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trip details only when user is actually ready to plan"""
    llm = get_llm(temperature=0.2)
    
    # Get conversation context to understand what they might be referring to
    recent_destinations = []
    chat_context = context.get('_chat_context', '')
    
    prompt = f"""Extract specific trip planning details from: "{user_input}"

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
Return JSON only.
"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        import json
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
            # Only return valid extractions
            valid_data = {}
            for key, value in extracted.items():
                if value and str(value).strip():
                    valid_data[key] = value
            return valid_data
    except Exception as e:
        print(f"Detail extraction failed: {e}")
    
    # Fallback: simple keyword extraction
    fallback_data = {}
    lower_input = user_input.lower()
    
    # Extract duration
    duration_match = re.search(r'(\d+)\s*days?', lower_input)
    if duration_match:
        fallback_data['duration'] = int(duration_match.group(1))
    
    # Extract travelers
    traveler_match = re.search(r'(\d+)\s*(?:people|person|travelers?)', lower_input)
    if traveler_match:
        fallback_data['travelers'] = int(traveler_match.group(1))
    
    # Extract origin city
    from_match = re.search(r'from\s+([A-Za-z\s]+?)(?:\s|$|,)', user_input, re.IGNORECASE)
    if from_match:
        fallback_data['origin'] = from_match.group(1).strip().title()
    
    return fallback_data

def conversation_agent(state: ConversationAgentState):
    """Modern, intelligent conversation agent that handles exploration and planning naturally"""
    
    user_input = state.get('user_input', '')
    chat_history = state.get('chat_history', []) or []
    context = state.get('context', {}) or {}
    
    # Keep track of conversation context for better understanding
    recent_mentions = []
    for msg in chat_history[-6:]:
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            content = msg.get('content', '')
            # Extract place names mentioned in recent responses
            place_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)
            recent_mentions.extend([p for p in place_matches if len(p) > 3])
    
    context['_recent_mentions'] = list(set(recent_mentions))
    context['_chat_context'] = ' '.join([msg.get('content', '')[:100] for msg in chat_history[-3:] if isinstance(msg, dict)])
    
    # Classify user intent with LLM
    intent_data = _classify_user_intent(user_input, chat_history)
    intent = intent_data.get('intent', 'chat')
    
    print(f"[DEBUG] User intent: {intent_data}")
    
    # Handle different intents
    if intent == 'explore':
        response = _handle_exploration(user_input, intent_data, context)
        return {
            'response': response,
            'context': context,
            'planning_stage': 'explore',
            'missing_info': False
        }
    
    elif intent == 'chat':
        response = _handle_general_chat(user_input, context)
        return {
            'response': response,
            'context': context,
            'planning_stage': 'chat',
            'missing_info': False
        }
    
    elif intent == 'plan':
        # Extract planning details
        extracted = _extract_planning_details(user_input, context)
        
        # Update context with extracted details
        for key, value in extracted.items():
            if key == 'origin':
                context['user_city'] = value
            elif key == 'date':
                context['start_date'] = value
            elif key == 'duration':
                context['duration_days'] = value
            elif key == 'travelers':
                context['number_of_travelers'] = value
            else:
                context[key] = value
        
        # Check if we have enough to start planning
        has_destination = bool(context.get('destination'))
        
        # Smart destination inference from recent conversation
        if not has_destination and context.get('_recent_mentions'):
            # If user says "plan a trip" after we discussed destinations, try to infer
            llm = get_llm(temperature=0.3)
            mentions = context['_recent_mentions'][:5]  # Limit to avoid token overflow
            
            prompt = f"""The user said: "{user_input}"

Recently mentioned places in our conversation: {mentions}

Are they likely referring to one of these places for their trip planning? If yes, which one? If unclear, return "unclear".

Return just the place name or "unclear".
"""
            
            try:
                inferred = llm.invoke(prompt).content.strip().strip('"\'')
                if inferred.lower() != 'unclear' and inferred in mentions:
                    context['destination'] = inferred
                    has_destination = True
                    print(f"[DEBUG] Inferred destination: {inferred}")
            except Exception:
                pass
        
        if not has_destination:
            # Ask them to specify destination
            response = "Which destination would you like me to plan for? I can help you choose from the places we discussed or somewhere completely new!"
            return {
                'response': response,
                'context': context,
                'planning_stage': 'plan',
                'missing_info': True
            }
        
        # Ready to hand off to trip planner
        return {
            'response': '',  # Trip planner will handle the response
            'context': context,
            'planning_stage': 'plan',
            'missing_info': False,
            'ready_for_planning': True
        }
    
    # Fallback
    return {
        'response': "I'm here to help you plan amazing trips! What kind of adventure are you thinking about?",
        'context': context,
        'planning_stage': 'chat',
        'missing_info': False
    }

def build_conversation_graph():
    """Build the conversation graph with intelligent routing"""
    graph = StateGraph(ConversationAgentState)
    
    graph.add_node('conversation', conversation_agent)
    graph.add_node('trip_planner', trip_planner_node)
    
    def smart_router(state):
        """Route based on user intent and readiness, not rigid slot checking"""
        
        # Only route to trip planner if user is explicitly ready to plan
        # and has at least a destination
        ready_for_planning = state.get('ready_for_planning', False)
        has_destination = bool(state.get('context', {}).get('destination'))
        
        if ready_for_planning and has_destination:
            return 'trip_planner'
        else:
            return END
    
    graph.add_conditional_edges(
        'conversation', 
        smart_router, 
        {'trip_planner': 'trip_planner', END: END}
    )
    
    graph.add_edge('trip_planner', END)
    graph.set_entry_point('conversation')
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# Cache the graph
_GRAPH_CACHE = None

def run_conversation_graph(user_input, chat_history, context=None):
    """Run the conversation with the new intelligent routing"""
    global _GRAPH_CACHE
    
    if _GRAPH_CACHE is None:
        _GRAPH_CACHE = build_conversation_graph()
    
    context = context or {}
    state: ConversationAgentState = {
        'user_input': user_input,
        'chat_history': chat_history,
        'context': context
    }
    
    # Get or create thread ID
    thread_id = context.get('_thread_id')
    if not thread_id:
        thread_id = 'default'
        context['_thread_id'] = thread_id
        state['context'] = context
    
    result = _GRAPH_CACHE.invoke(state, config={'configurable': {'thread_id': thread_id}})
    return result
