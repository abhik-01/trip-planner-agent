from tools.activity import get_activity_tool
from tools.flight import get_flight_tool, format_flights_for_display
from tools.map import get_map_tool
from tools.weather import get_weather_tool
from tools.assembler import get_assembler_tool

from tools.budget import get_budget_tool
from classTypes.class_types import TripPlannerState
from utils.set_llm import get_llm

from datetime import datetime
import re


def trip_planner_node(state: TripPlannerState):
    """Core trip planning node: ensures required fields, calls tools, assembles itinerary."""
    ctx = state.get('context', {}) or {}
    chat_history = state.get('chat_history', []) or []
    stage = state.get('planning_stage') or 'plan'

    # If still exploring, do not run heavy tools yet.
    if stage != 'plan':
        return {
            'response': '',
            'missing_info': False,
            'context': ctx,
            'planning_stage': stage
        }

    print(f"[DEBUG] trip_planner_node context: {ctx}")

    # Build simple history string (not used for chit-chat, but could help future reasoning)
    history_str = ""
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'user':
                history_str += f"User: {content}\n"
            elif role == 'assistant':
                history_str += f"Agent: {content}\n"
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            history_str += f"User: {msg[0]}\nAgent: {msg[1]}\n"

    # Month normalization helper: convert month name to first-day-of-month (current year)
    def normalize_month(value: str) -> str:
        if not value:
            return value
        v = value.strip()
        # Already ISO date?
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            return v
        try:
            dt = datetime.strptime(v.lower(), "%B")
            return f"{datetime.utcnow().year}-{dt.month:02d}-01"
        except Exception:
            return value

    # If user just provided an ISO date in the last utterance and start_date missing, capture it
    if not ctx.get('start_date'):
        # Restore from committed if previously stored
        if ctx.get('committed_start_date'):
            ctx['start_date'] = ctx['committed_start_date']
        else:
            ui = state.get('user_input','')
            m_date = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', ui)
            if m_date:
                ctx['start_date'] = m_date.group(1)
            else:
                # Accept phrases like 'use that' to commit proposed_start_date
                if ctx.get('proposed_start_date') and re.search(r'\b(use that|that works|ok|okay|yes)\b', ui.lower()):
                    ctx['start_date'] = ctx['proposed_start_date']

    if ctx.get('start_date') and len(str(ctx['start_date']).split()) == 1:
        ctx['start_date'] = normalize_month(str(ctx['start_date']))

    # Support alias 'departure_city'
    if ctx.get('departure_city') and not ctx.get('user_city'):
        ctx['user_city'] = ctx['departure_city']

    # If start_date exists and is ISO, mark committed
    if ctx.get('start_date') and re.match(r'^\d{4}-\d{2}-\d{2}$', str(ctx['start_date'])):
        ctx['committed_start_date'] = ctx['start_date']
    # If start_date is a plain month token, move to proposed_month and treat as missing
    elif ctx.get('start_date') and not re.match(r'^\d{4}-\d{2}-\d{2}$', str(ctx['start_date'])):
        month_token = str(ctx['start_date']).lower().strip()
        if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)', month_token):
            ctx['proposed_month'] = month_token
            ctx.pop('start_date', None)

    required = ['user_city', 'destination', 'start_date']
    # Treat committed_start_date as satisfying start_date
    if ctx.get('committed_start_date') and not ctx.get('start_date'):
        ctx['start_date'] = ctx['committed_start_date']
    missing = [f for f in required if not ctx.get(f)]
    print(f"[DEBUG] trip_planner_node missing fields: {missing}")
    if missing:
        prompts = {
            'user_city': "To find flights, what's your departure city?",
            'destination': "Where would you like to travel?",
            'start_date': "What is your departure date? (YYYY-MM-DD) If you want a recommendation, ask me to suggest the best date."
        }
        # Special handling for month-only
        if 'start_date' in missing:
            month_map = {'jan':'01','january':'01','feb':'02','february':'02','mar':'03','march':'03','apr':'04','april':'04','may':'05','jun':'06','june':'06','jul':'07','july':'07','aug':'08','august':'08','sep':'09','sept':'09','september':'09','oct':'10','october':'10','nov':'11','november':'11','dec':'12','december':'12'}
            proposed_month = ctx.get('proposed_month')
            if not proposed_month:
                # Attempt detect from last user input
                last = state.get('user_input','').lower()
                for tk in month_map.keys():
                    if re.search(rf'\b{tk}\b', last):
                        proposed_month = tk
                        break
            if proposed_month:
                mm = month_map.get(proposed_month, None)
                if mm:
                    year = datetime.utcnow().year
                    suggested = f"{year}-{mm}-15"
                    ctx['proposed_start_date'] = suggested
                    return {"response": f"You mentioned {proposed_month}. A good date could be {suggested}. Provide a specific date (YYYY-MM-DD) or say 'use that'.", "missing_info": True, "context": ctx}
        for field in required:
            if field in missing:
                return {"response": prompts[field], "missing_info": True, "context": ctx}

    # Prepare destination vars early (needed for duration suggestion heuristics)
    destination_full = ctx.get('destination')
    # Ask for duration if absent
    if not ctx.get('duration_days'):
        user_last = state.get('user_input','').lower().strip()
        # If user replies affirmatively after a proposal, auto-accept
        if ctx.get('proposed_duration') and re.fullmatch(r'(yes|ok|okay|sure|fine|that works|sounds good|go ahead|yep|yeah)\.?', user_last):
            ctx['duration_days'] = ctx['proposed_duration']
        else:
            # Heuristic recommendation
            pref_map = [
                (['weekend','short','quick','2 day','2-day','couple','few'], 3),
                (['mid','medium','moderate','balanced'], 5),
                (['week','7 day','7-day'], 7),
                (['long','extended','10 day','10-day'], 8),
            ]
            suggested = ctx.get('proposed_duration')
            if suggested is None:
                for keys, val in pref_map:
                    if any(k in user_last for k in keys):
                        suggested = val
                        break
            dest_lower = (destination_full or '').lower()
            if suggested is None:
                if any(k in dest_lower for k in ['trek','himalaya','patagonia','safari']):
                    suggested = 7
                elif any(k in dest_lower for k in ['city','capital','paris','london','tokyo']):
                    suggested = 5
                elif any(k in dest_lower for k in ['beach','island']):
                    suggested = 5
            if suggested is None:
                suggested = 5
            ctx['proposed_duration'] = suggested
            return {"response": f"How many days do you want this trip to last? I suggest {suggested} days. Reply with a number or just say 'ok' to accept.", "missing_info": True, "context": ctx}

    # Ask for traveler count if absent
    if not ctx.get('number_of_travelers'):
        return {"response": "How many travelers are going? (e.g., 2 people)", "missing_info": True, "context": ctx}

    # All required fields present – proceed to tools
    destination = destination_full.split(',')[0].strip() if isinstance(destination_full, str) else destination_full
    start_date = ctx.get('start_date')
    user_city = ctx.get('user_city')
    travelers = ctx.get('number_of_travelers') or 1
    if isinstance(travelers, str):
        mt = re.search(r'\d+', travelers)
        if mt:
            try:
                travelers = int(mt.group(0))
            except Exception:
                travelers = 1
    ctx['number_of_travelers'] = travelers
    duration_days = ctx.get('duration_days') or 3

    def safe_call(label, fn, *args, **kwargs):
        try:
            print(f"[DEBUG] TOOL_CALL {label} args={args} kwargs={kwargs}")
            out = fn(*args, **kwargs)
            if hasattr(out, 'content'):
                return out.content
            return out
        except Exception as e:
            print(f"[ERROR] tool {label} failed: {e}")
            return ""

    results = {}

    # Activities
    if destination_full:
        act_tool = get_activity_tool()
        results['activities'] = safe_call('activities', act_tool.func, destination_full)
    else:
        results['activities'] = ""

    # Map / Nearby
    if destination_full:
        map_tool = get_map_tool()
        interests = ctx.get('interests') or ['tourism']
        # choose first interest as category string
        category = interests[0] if isinstance(interests, list) and interests else 'tourism'
        results['map'] = safe_call('map', map_tool.func, destination_full, category)
    else:
        results['map'] = ""

    # Weather
    if destination and start_date:
        weather_tool = get_weather_tool()
        results['weather'] = safe_call('weather', weather_tool.func, destination, start_date)
    else:
        results['weather'] = ""

    # Flights
    if user_city and destination and start_date:
        flight_tool = get_flight_tool()
        flights_raw = safe_call('flights', flight_tool.func, user_city, destination, start_date, travelers)
        if flights_raw:
            try:
                formatted = format_flights_for_display(flights_raw)
            except Exception as fe:
                print(f"[WARN] format flights failed: {fe}")
                formatted = ""
            # If user currency requested (and not INR already), append converted line per flight using currency tool
            user_currency = ctx.get('user_currency')
            if user_currency and user_currency.upper() not in ['USD', 'INR'] and isinstance(flights_raw, list) and flights_raw and 'error' not in flights_raw[0]:
                try:
                    from tools.currency import convert_amount
                    extra_lines = []
                    for f in flights_raw:
                        if isinstance(f, dict) and 'price' in f and 'currency' in f:
                            amt = f['price']
                            cur = f['currency']
                            if isinstance(amt, (int,float)):
                                conv = convert_amount(float(amt), cur, user_currency.upper())
                                extra_lines.append(f"Converted: {user_currency.upper()} {conv} (from {cur} {amt})")
                    if extra_lines:
                        formatted += "\n" + "\n".join(extra_lines)
                except Exception as e:
                    print(f"[WARN] flight currency annotate failed: {e}")
            results['flights'] = formatted
            # derive cheapest price for budget
            if isinstance(flights_raw, list) and flights_raw and 'price' in flights_raw[0] and 'error' not in flights_raw[0]:
                try:
                    prices = []
                    for f in flights_raw:
                        p = f.get('price') if isinstance(f, dict) else None
                        if isinstance(p, (int, float)):
                            prices.append(p)
                        elif isinstance(p, str):
                            try:
                                prices.append(float(p.split()[0]))
                            except Exception:
                                pass
                    if prices:
                        ctx['flight_cost'] = min(prices)
                except Exception as e:
                    print(f"[WARN] derive flight cost failed: {e}")
        else:
            results['flights'] = ""
    else:
        results['flights'] = ""

    # Budget (best-effort)
    try:
        budget_tool = get_budget_tool()
        # minimal trip_details for budget
        trip_details = {
            'destination': destination,
            'flight_cost': ctx.get('flight_cost', ''),
            'nights': max(duration_days - 1, 1),
            'travelers': travelers,
            'activities': [],
            'days': duration_days
        }
        budget_raw = safe_call('budget', budget_tool.func, trip_details)
        budget_text = str(getattr(budget_raw, 'content', budget_raw)) if budget_raw else ''
        # Determine target currency (respect inferred INR)
        user_currency = ctx.get('user_currency')
        if not user_currency:
            # Infer INR if destination or user_city indicates India
            txt = f"{destination} {user_city}".lower()
            if any(ind in txt for ind in [' india',' goa',' mumbai',' delhi',' kolkata',' bengaluru',' bangalore',' chennai',' hyderabad',' pune',' jaipur',' kerala']):
                user_currency = 'INR'
                ctx['user_currency'] = 'INR'
        if user_currency and user_currency.upper() != 'USD':
            try:
                from tools.currency import convert_amount
                # naive extraction of total USD figure
                m = re.search(r'(\$)(\d+[\d\.]*)', budget_text)
                if m:
                    usd_value = float(m.group(2))
                    converted = convert_amount(usd_value, 'USD', user_currency.upper())
                    budget_text += f"\nApprox total in {user_currency.upper()}: {converted} {user_currency.upper()}"
            except Exception as e:
                print(f"[WARN] budget currency convert failed: {e}")
        results['budget'] = budget_text
    except Exception as e:
        print(f"[WARN] budget tool failed: {e}")
        results['budget'] = ''

    # Assemble itinerary
    try:
        assembler = get_assembler_tool()
        itinerary_raw = assembler.func({**ctx, **results})
        if hasattr(itinerary_raw, 'content'):
            itinerary = itinerary_raw.content
        else:
            itinerary = str(itinerary_raw)
    except Exception as e:
        print(f"[ERROR] assembler failed: {e}")
        itinerary = "Here is your trip plan (partial)."

    state.update({
        "response": itinerary,
        "missing_info": False,
        "context": ctx
    })
    # Attach tool outputs for potential UI debugging
    for k,v in results.items():
        state[k] = v
    return state
