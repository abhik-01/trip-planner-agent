from classTypes.class_types import TripPlannerState
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import format_prompt, PromptType
from tools.activity import get_activity_tool
from tools.flight import get_flight_tool
from tools.map import get_map_tool
from tools.weather import get_weather_tool
from tools.assembler import get_assembler_tool
from tools.budget import get_budget_tool
from utils.set_llm import get_llm
from utils.safety import (
    validate_response_safety, 
    is_sensitive_destination
)


def trip_planner_node(state: TripPlannerState):
    """Simplified trip planning node - only runs when user is ready to plan"""
    ctx = state.get('context', {}) or {}
    
    # Extract basic requirements
    destination = ctx.get('destination')
    user_city = ctx.get('user_city')
    start_date = ctx.get('start_date') or ctx.get('committed_start_date')
    
    # If we're missing critical info, ask for it naturally
    if not destination:
        return {
            "response": "Which destination would you like me to plan for?",
            "missing_info": True,
            "context": ctx
        }
    
    if not user_city:
        return {
            "response": f"Great choice on {destination}! Which city will you be departing from?",
            "missing_info": True,
            "context": ctx
        }
    
    # Ask for missing details with intelligent defaults
    travelers = ctx.get('number_of_travelers')
    if not travelers:
        return {
            "response": "How many travelers will be going on this trip?",
            "missing_info": True,
            "context": ctx
        }
    
    duration = ctx.get('duration_days')
    if not duration:
        # Intelligent duration suggestion based on destination
        llm = get_llm()
        prompt = format_prompt(
            PromptType.DURATION_SUGGESTION,
            destination=destination
        )
        
        try:
            suggestion = llm.invoke(prompt).content.strip()
            return {
                "response": f"{suggestion}. How many days would you prefer?",
                "missing_info": True,
                "context": ctx
            }
        except Exception:
            return {
                "response": "How many days would you like this trip to last?",
                "missing_info": True,
                "context": ctx
            }
    
    # NEW: Ask for travel date if missing
    if not start_date:
        # Intelligent date suggestion based on destination
        llm = get_llm()
        prompt = format_prompt(
            PromptType.BEST_TIME_SUGGESTION,
            destination=destination
        )
        
        try:
            suggestion = llm.invoke(prompt).content.strip()
            return {
                "response": f"When would you like to travel? {suggestion}. Please provide your preferred departure date (YYYY-MM-DD format).",
                "missing_info": True,
                "context": ctx
            }
        except Exception:
            return {
                "response": "When would you like to travel? Please provide your preferred departure date (YYYY-MM-DD format).",
                "missing_info": True,
                "context": ctx
            }
    
    # Now we have enough info to plan - run tools in parallel for better performance
    try:
        results = {}
        
        def run_activity_tool():
            if destination:
                activity_tool = get_activity_tool()
                return ('activities', activity_tool.func(destination))
            return ('activities', None)
        
        def run_weather_tool():
            if destination and start_date:
                weather_tool = get_weather_tool()
                return ('weather', weather_tool.func(destination, start_date))
            return ('weather', None)
        
        def run_flight_tool():
            if user_city and destination and start_date:
                flight_tool = get_flight_tool()
                return ('flights', flight_tool.func(user_city, destination, start_date, travelers))
            return ('flights', None)
        
        def run_map_tool():
            if destination:
                map_tool = get_map_tool()
                return ('nearby', map_tool.func(destination))
            return ('nearby', None)
        
        def run_budget_tool():
            if destination and duration:
                budget_tool = get_budget_tool()
                budget_input = {
                    'destination': destination,
                    'nights': duration - 1,
                    'travelers': travelers,
                    'activities': ['general tourism'],
                    'days': duration
                }
                # Add flight cost and currency if available from context
                flight_cost = ctx.get('flight_cost')
                flight_currency = ctx.get('flight_currency', 'USD')
                if flight_cost:
                    budget_input['flight_cost'] = flight_cost
                    budget_input['flight_currency'] = flight_currency
                    
                return ('budget', budget_tool.func(budget_input))
            return ('budget', None)
        
        # Execute tools in parallel for independent operations        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all independent tool calls
            future_to_tool = {
                executor.submit(run_activity_tool): 'activities',
                executor.submit(run_weather_tool): 'weather', 
                executor.submit(run_map_tool): 'nearby'
            }
            
            # Flight and budget tools might depend on each other, but can run independently of others
            if user_city and destination and start_date:
                future_to_tool[executor.submit(run_flight_tool)] = 'flights'
            
            if destination and duration:
                future_to_tool[executor.submit(run_budget_tool)] = 'budget'
            
            # Collect results as they complete
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    key, result = future.result(timeout=15)  # 15s timeout per tool
                    if result:
                        results[key] = result
                except Exception:
                    # Continue with other tools even if one fails
                    pass
        
        # Extract flight cost and currency for budget calculations
        if 'flights' in results:
            try:
                flights_data = results['flights']
                if isinstance(flights_data, list) and flights_data and 'price' in flights_data[0]:
                    # Get the cheapest flight for budget calculation
                    cheapest_flight = min(flights_data, key=lambda x: x.get('price', float('inf')))
                    ctx['flight_cost'] = cheapest_flight.get('price')
                    ctx['flight_currency'] = cheapest_flight.get('currency', 'USD')
            except Exception:
                pass

        # Store all tool results in context for later reference
        ctx['tool_results'] = results
        ctx['last_trip_data'] = {
            'destination': destination,
            'user_city': user_city,
            'start_date': start_date,
            'duration_days': duration,
            'number_of_travelers': travelers,
            **results
        }
        
        # Check for critical failures and inform user upfront
        critical_issues = []
        
        # Check flight search failures
        if 'flights' in results:
            flights_data = results['flights']
            if isinstance(flights_data, list) and flights_data and 'error' in flights_data[0]:
                critical_issues.append(f"✈️ **Flight Search**: {flights_data[0]['error']}")
        
        # Check weather failures 
        if 'weather' in results:
            weather_data = results['weather']
            if isinstance(weather_data, str) and ('could not find' in weather_data.lower() or 'error' in weather_data.lower()):
                critical_issues.append(f"🌤️ **Weather**: I couldn't get weather data for {destination}")
        
        # Assemble everything into a coherent itinerary
        assembler_tool = get_assembler_tool()
        trip_data = {
            **ctx,
            **results,
            'destination': destination,
            'user_city': user_city,
            'start_date': start_date,
            'duration_days': duration,
            'number_of_travelers': travelers
        }
        
        itinerary = assembler_tool.func(trip_data)
        
        # Add critical issues notification to the response if any
        if critical_issues:
            issues_text = "\n".join(critical_issues)
            final_response = f"🌟 **Your {destination} Trip Plan** 🌟\n\n**⚠️ Important Notes:**\n{issues_text}\n\n{itinerary}\n\n💡 **Need Help?** Feel free to ask about any specific aspect of your trip or let me know if you'd like me to search for alternatives!"
        else:
            final_response = itinerary
        
        # SAFETY CHECK: Validate final response for responsible travel advice
        context_info = f"Trip planning for {destination} from {user_city} for {travelers} travelers"
        response_safety = validate_response_safety(final_response, context_info)
        
        if not response_safety.get('is_safe', True):
            improved_response = response_safety.get('improved_response', '')
            if improved_response and improved_response.strip():
                final_response = improved_response

        # Add extra safety warnings for sensitive destinations
        if is_sensitive_destination(destination):
            safety_disclaimer = f"\n\n🛡️ **Important Safety Notice**: {destination} may require special precautions. Please check current travel advisories, local laws, and safety conditions before traveling. Consider consulting your country's travel advisory services."
            final_response += safety_disclaimer

        return {
            "response": final_response,
            "missing_info": False,
            "context": ctx,
            "planning_stage": "completed",
            "safety_validated": True
        }

    except Exception as e:
        return {
            "response": f"I encountered an issue while planning your trip. Let me try to help you manually - could you please provide your destination, departure city, travel dates, and number of travelers?",
            "missing_info": True,
            "context": ctx
        }
