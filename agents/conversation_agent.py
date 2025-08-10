from agents.trip_planner_agent import trip_planner_node
from classTypes.class_types import TripPlannerState
from concurrent.futures import ThreadPoolExecutor
from json import loads
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from prompts import format_prompt, PromptType
from re import search, DOTALL
from typing import Dict, Any, List
from tools.destination import get_destination_tool
from utils.intelligent_intent import IntelligentIntentClassifier, SemanticQueryClassifier
from utils.set_llm import get_llm
from utils.safety import (
    screen_user_input_safety, 
    validate_response_safety, 
    get_safety_refusal_response
)


ConversationAgentState = TripPlannerState


def _classify_user_intent(user_input: str, chat_history: List[Dict[str,str]]) -> Dict[str, Any]:
    """Use intelligent LLM-based intent classification instead of hardcoded keywords"""
    return IntelligentIntentClassifier.classify_intent(user_input, chat_history)

def _handle_exploration(user_input: str, intent_data: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Handle exploration requests - suggestions, browsing, etc. with async optimization"""
    
    exploring = (intent_data.get('exploring') or '').lower()
    
    if 'destination' in exploring or 'places' in exploring or not exploring:
        # Use destination suggestion tool with parallel LLM processing
        try:
            def get_suggestions():
                dest_tool = get_destination_tool()
                
                # Enhanced input with context awareness
                enhanced_input = user_input
                
                # Add context from recent conversation if this is a follow-up request
                is_followup = context.get('_is_followup_request', False)
                if is_followup:
                    # Use available context info
                    recent_context = context.get('_chat_context', '')
                    geographic_context = context.get('_geographic_context', [])
                    temporal_context = context.get('_temporal_context', [])
                    geographic_constraints = context.get('_geographic_constraints', [])
                    follow_up_context = context.get('_follow_up_context', '')
                    
                    # Build comprehensive context with priority on geographic constraints
                    context_parts = []
                    if geographic_constraints:
                        context_parts.append(f"GEOGRAPHIC CONSTRAINT: {', '.join(geographic_constraints[:3])}")
                    if recent_context:
                        context_parts.append(recent_context[:200])
                    if geographic_context:
                        context_parts.append(f"Geographic focus: {', '.join(geographic_context[:3])}")
                    if temporal_context:
                        context_parts.append(f"Time context: {', '.join(temporal_context[:2])}")
                    if follow_up_context:
                        context_parts.append(follow_up_context)
                    
                    if context_parts:
                        enhanced_input = f"{' '.join(context_parts)} - User request: {user_input}"
                
                return dest_tool.func(enhanced_input)
            
            def get_context_response():
                llm = get_llm()
                prompt = format_prompt(
                    PromptType.EXPLORATION_INTRO,
                    user_input=user_input
                )
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
            return "I'd love to suggest some amazing destinations! Could you tell me what kind of experience you're looking for?"
    
    # For other types of exploration, use LLM
    llm = get_llm()
    prompt = format_prompt(
        PromptType.EXPLORATION_GENERAL,
        user_input=user_input
    )
    
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return "That sounds exciting! What kind of travel experience are you looking for?"

def _handle_general_chat(user_input: str, context: Dict[str, Any]) -> str:
    """Handle general conversation, greetings, and any non-travel topics intelligently"""
    
    # Check if user is asking about specific aspects of their previous trip planning
    if _is_trip_specific_inquiry(user_input, context):
        return _handle_trip_specific_inquiry(user_input, context)
    
    llm = get_llm()
    
    # Enhanced prompt to handle any topic and naturally steer to travel
    prompt = format_prompt(
        PromptType.GENERAL_CHAT,
        user_input=user_input
    )
    
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return "Hello! I'm your travel planning assistant. I'm here to help you discover amazing destinations and plan unforgettable trips. What kind of adventure interests you?"


def _is_trip_specific_inquiry(user_input: str, context: Dict[str, Any]) -> bool:
    """Use semantic classification to detect trip-specific inquiries"""
    
    # Must have previous trip data to answer specific questions
    if not context.get('last_trip_data') or not context.get('tool_results'):
        return False
    
    # Use semantic classifier to determine if this is a travel-related query
    query_type = SemanticQueryClassifier.classify_travel_query(user_input, context.get('last_trip_data', {}))
    
    # Any specific travel query type (not 'general') indicates trip-specific inquiry
    return query_type != 'general'


def _handle_trip_specific_inquiry(user_input: str, context: Dict[str, Any]) -> str:
    """Handle specific questions about different aspects of the planned trip using semantic classification"""
    
    last_trip_data = context.get('last_trip_data', {})
    tool_results = context.get('tool_results', {})

    # Use semantic classifier to determine the specific type of travel query
    query_type = SemanticQueryClassifier.classify_travel_query(user_input, last_trip_data)
    
    # Route to appropriate handler based on semantic classification
    if query_type == 'weather':
        return _handle_weather_inquiry(user_input, last_trip_data, tool_results)
    elif query_type == 'activities':
        return _handle_activity_inquiry(user_input, last_trip_data, tool_results)
    elif query_type == 'nearby':
        return _handle_nearby_inquiry(user_input, last_trip_data, tool_results)
    elif query_type == 'budget':
        return _handle_budget_inquiry(user_input, last_trip_data, tool_results)
    elif query_type == 'flights':
        return _handle_flight_inquiry(user_input, context)
    elif query_type in ['accommodation', 'food']:
        # Handle accommodation and food queries with general trip inquiry for now
        return _handle_general_trip_inquiry(user_input, last_trip_data, tool_results)
    else:
        # For general queries or unclassified, use the general trip inquiry handler
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
    prompt = format_prompt(
        PromptType.WEATHER_INQUIRY,
        user_input=user_input,
        destination=destination,
        start_date=start_date or 'Not specified',
        weather_data=weather_data
    )
    
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
    prompt = format_prompt(
        PromptType.ACTIVITY_INQUIRY,
        user_input=user_input,
        destination=destination,
        activities_data=activities_data
    )
    
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
    prompt = format_prompt(
        PromptType.NEARBY_INQUIRY,
        user_input=user_input,
        destination=destination,
        nearby_data=nearby_data
    )
    
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
    prompt = format_prompt(
        PromptType.BUDGET_INQUIRY,
        user_input=user_input,
        destination=destination,
        duration=duration,
        travelers=travelers,
        budget_data=budget_data
    )
    
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
    
    prompt = format_prompt(
        PromptType.GENERAL_TRIP_INQUIRY,
        user_input=user_input,
        trip_summary=trip_summary
    )
    
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
    
    prompt = format_prompt(
        PromptType.FLIGHT_INQUIRY,
        user_input=user_input,
        flight_summary=flight_summary
    )
    
    try:
        response = llm.invoke(prompt)
        return str(getattr(response, 'content', response))
    except Exception:
        # Fallback to structured response
        cheapest_flight = min(flights_data, key=lambda x: x.get('price', float('inf')))
        return f"I found {len(flights_data)} flight options for your {destination} trip from {user_city}. The cheapest option is {cheapest_flight.get('airline', 'N/A')} at ₹{cheapest_flight.get('price_in_inr', 'N/A')} INR. Would you like more details about the flight options?"

def _extract_planning_details(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trip details using intelligent LLM-based extraction"""
    llm = get_llm(temperature=0.2)
    
    # Get conversation context to understand what they might be referring to
    chat_context = context.get('_chat_context', '')
    
    prompt = format_prompt(
        PromptType.PLANNING_DETAILS_EXTRACTION,
        user_input=user_input,
        chat_context=chat_context,
        context=context
    )
    
    try:
        response = llm.invoke(prompt).content.strip()
        json_match = search(r'\{.*\}', response, DOTALL)

        if json_match:
            extracted = loads(json_match.group())

            # Only return valid extractions
            valid_data = {}

            for key, value in extracted.items():
                if value and str(value).strip():
                    valid_data[key] = value

            return valid_data
    except Exception as e:    
        # Enhanced LLM fallback for robust extraction
        try:
            prompt = format_prompt(
                PromptType.FALLBACK_DETAIL_EXTRACTION,
                user_input=user_input
            )
            
            fallback_response = llm.invoke(prompt).content.strip()
            json_match = search(r'\{.*\}', fallback_response, DOTALL)

            if json_match:
                return loads(json_match.group())
        except Exception:
            pass
    
    return {}

def conversation_agent(state: ConversationAgentState):
    """Modern, intelligent conversation agent that handles exploration and planning naturally"""
    
    user_input = state.get('user_input', '')
    chat_history = state.get('chat_history', []) or []
    context = state.get('context', {}) or {}
    
    # SAFETY CHECK: Screen user input first
    safety_check = screen_user_input_safety(user_input)

    if not safety_check.get('is_safe', True):
        concern_type = safety_check.get('concern_type', 'unknown')

        # Return appropriate refusal response
        refusal_response = get_safety_refusal_response(
            concern_type, 
            safety_check.get('suggested_response', '')
        )
        
        return {
            'response': refusal_response,
            'context': context,
            'planning_stage': 'safety_refusal',
            'missing_info': False,
            'safety_concern': concern_type
        }
    
    # Continue with normal conversation processing...
    
    # Enhanced LLM-based context extraction for better understanding
    llm = get_llm(temperature=0.3)
    
    # Extract context from recent conversation using LLM
    recent_msgs = [msg.get('content', '')[:200] for msg in chat_history[-6:] if isinstance(msg, dict)]
    recent_conversation = ' '.join(recent_msgs)
    
    if recent_conversation.strip():
        prompt = format_prompt(
            PromptType.SEMANTIC_CONTEXT_EXTRACTION,
            recent_conversation=recent_conversation
        )
        
        try:
            context_response = llm.invoke(prompt).content.strip()
            json_match = search(r'\{.*\}', context_response, DOTALL)

            if json_match:
                extracted_context = loads(json_match.group())
                context['_recent_mentions'] = extracted_context.get('recent_mentions', [])
                context['_geographic_context'] = extracted_context.get('geographic_context', [])
                context['_temporal_context'] = extracted_context.get('temporal_context', [])
                context['_geographic_constraints'] = extracted_context.get('geographic_constraints', [])
        except Exception as e:
            # Fallback to simple extraction
            context['_recent_mentions'] = []
            context['_geographic_context'] = []
            context['_temporal_context'] = []
            context['_geographic_constraints'] = []
    
    # Enhanced chat context with geographic and temporal awareness
    context['_chat_context'] = ' '.join(recent_msgs[-3:])
    
    # Add specific context for follow-up requests using semantic understanding
    llm = get_llm(temperature=0.3)
    
    # Check if this is a follow-up request semantically
    prompt = format_prompt(
        PromptType.SEMANTIC_FOLLOWUP_DETECTION,
        user_input=user_input
    )
    
    try:
        is_followup = llm.invoke(prompt).content.strip().lower() == 'yes'
        context['_is_followup_request'] = is_followup
        if is_followup:
            geo_context = context.get('_geographic_context', [])
            geo_constraints = context.get('_geographic_constraints', [])
            time_context = context.get('_temporal_context', [])
            
            follow_up_parts = []
            if geo_constraints:
                follow_up_parts.append(f"Geographic constraint: {', '.join(geo_constraints[:2])}")
            elif geo_context:
                follow_up_parts.append(f"Previous context: {', '.join(geo_context[:3])}")
            if time_context:
                follow_up_parts.append(f"Time context: {', '.join(time_context[:2])}")
            
            if follow_up_parts:
                context['_follow_up_context'] = ' '.join(follow_up_parts)
    except Exception:
        # Fallback to simple keyword check
        is_followup = any(word in user_input.lower() for word in ['more', 'few more', 'other', 'additional', 'alternative'])
        context['_is_followup_request'] = is_followup
        if is_followup:
            geo_context = context.get('_geographic_context', [])
            geo_constraints = context.get('_geographic_constraints', [])
            time_context = context.get('_temporal_context', [])
            
            follow_up_parts = []
            if geo_constraints:
                follow_up_parts.append(f"Geographic constraint: {', '.join(geo_constraints[:2])}")
            elif geo_context:
                follow_up_parts.append(f"Previous context: {', '.join(geo_context[:3])}")
            if time_context:
                follow_up_parts.append(f"Time context: {', '.join(time_context[:2])}")
            
            if follow_up_parts:
                context['_follow_up_context'] = ' '.join(follow_up_parts)
    
    # Classify user intent with LLM
    intent_data = _classify_user_intent(user_input, chat_history)
    intent = intent_data.get('intent', 'chat')
    
    # Handle different intents
    response = ""
    if intent == 'explore':
        response = _handle_exploration(user_input, intent_data, context)
    elif intent == 'chat':
        response = _handle_general_chat(user_input, context)
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
            
            prompt = format_prompt(
                PromptType.DESTINATION_INFERENCE,
                user_input=user_input,
                mentions=mentions
            )
            
            try:
                inferred = llm.invoke(prompt).content.strip().strip('"\'')
                if inferred.lower() != 'unclear' and inferred in mentions:
                    context['destination'] = inferred
                    has_destination = True
            except Exception:
                pass
        
        if not has_destination:
            # Ask them to specify destination
            response = "Which destination would you like me to plan for? I can help you choose from the places we discussed or somewhere completely new!"
        else:
            # Ready to hand off to trip planner
            return {
                'response': '',  # Trip planner will handle the response
                'context': context,
                'planning_stage': 'plan',
                'missing_info': False,
                'ready_for_planning': True
            }
    
    # Fallback
    if not response:
        response = "I'm here to help you plan amazing trips! What kind of adventure are you thinking about?"
    
    # SAFETY CHECK: Validate response before returning
    response_safety = validate_response_safety(response, user_input)
    
    if not response_safety.get('is_safe', True):
        improved_response = response_safety.get('improved_response', '')
        if improved_response and improved_response.strip():
            response = improved_response
    
    return {
        'response': response,
        'context': context,
        'planning_stage': 'chat' if intent == 'chat' else 'explore',
        'missing_info': False,
        'safety_validated': True
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
