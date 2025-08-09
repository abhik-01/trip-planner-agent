from tools.activity import get_activity_tool
from tools.flight import get_flight_tool
from tools.map import get_map_tool
from tools.weather import get_weather_tool
from tools.assembler import get_assembler_tool
from tools.budget import get_budget_tool
from classTypes.class_types import TripPlannerState
from utils.set_llm import get_llm

from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def trip_planner_node(state: TripPlannerState):
    """Simplified trip planning node - only runs when user is ready to plan"""
    ctx = state.get('context', {}) or {}
    user_input = state.get('user_input', '')
    
    print(f"[DEBUG] trip_planner_node context: {ctx}")
    
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
        prompt = f"""Suggest an ideal trip duration for {destination} in 3-7 words. Consider typical tourist activities and travel distance. Just state the recommendation naturally, like "I'd recommend 5 days" or "A week would be perfect"."""
        
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
        prompt = f"""What is the best time of year to visit {destination}? Consider weather, crowds, and local events. Suggest the best months and explain why in 2-3 sentences."""
        
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
        print(f"[DEBUG] Running tools in parallel for {destination}")
        start_time = time.time()
        
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
                        print(f"[DEBUG] ✅ {tool_name} completed")
                except Exception as e:
                    print(f"[DEBUG] ❌ {tool_name} failed: {e}")
                    # Continue with other tools even if one fails
                    
        execution_time = time.time() - start_time
        print(f"[DEBUG] Parallel tool execution completed in {execution_time:.2f}s")
        
        # Extract flight cost and currency for budget calculations
        if 'flights' in results:
            try:
                flights_data = results['flights']
                if isinstance(flights_data, list) and flights_data and 'price' in flights_data[0]:
                    # Get the cheapest flight for budget calculation
                    cheapest_flight = min(flights_data, key=lambda x: x.get('price', float('inf')))
                    ctx['flight_cost'] = cheapest_flight.get('price')
                    ctx['flight_currency'] = cheapest_flight.get('currency', 'USD')
                    print(f"[DEBUG] Extracted flight cost: {ctx['flight_cost']} {ctx['flight_currency']}")
            except Exception as e:
                print(f"[DEBUG] Failed to extract flight cost: {e}")
        
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
        
        return {
            "response": itinerary,
            "missing_info": False,
            "context": ctx,
            "planning_stage": "completed"
        }
        
    except Exception as e:
        print(f"[ERROR] Trip planning failed: {e}")
        return {
            "response": f"I encountered an issue while planning your trip. Let me try to help you manually - could you please provide your destination, departure city, travel dates, and number of travelers?",
            "missing_info": True,
            "context": ctx
        }
