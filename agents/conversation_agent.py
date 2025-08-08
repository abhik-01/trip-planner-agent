from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, List
import re, asyncio, time
from datetime import date

from tools.destination import get_destination_tool
from utils.set_llm import get_llm
from classTypes.class_types import TripPlannerState
from utils.plan import build_plan, replan
from utils.tooling import run_tool, validate_flights, validate_weather, extract_min_price
from utils.models import PlanStep

ConversationAgentState = TripPlannerState

ALLOWED_CONTEXT_KEYS = {
    'destination','start_date','end_date','number_of_travelers','interests','user_city','budget',
    'proposed_start_date','user_currency','duration_days','committed_start_date'
}

SLOT_ORDER = [
    'user_city',
    'destination',
    'start_date',
    'duration_days',
    'number_of_travelers'
]

def _history_to_text(chat_history: List[Dict[str,str]]):
    lines = []
    for msg in chat_history[-12:]:
        if isinstance(msg, dict):
            r = msg.get('role'); c = msg.get('content')
            if r and c:
                lines.append(f"{r.capitalize()}: {c}")
    return "\n".join(lines)

def _classify_intent(user_input: str, history: str) -> str:
    prompt = f"""Classify the user's primary intent into exactly one label: SMALLTALK, EXPLORE, or PLAN.
SMALLTALK: greetings, chit chat, general non-planning questions.
EXPLORE: asking for destination ideas, vague preferences, browsing.
PLAN: providing/asking for concrete trip logistics (dates, origin city, budget, flights, itinerary) OR selecting a destination to proceed.
History (recent):\n{history}\nUser: {user_input}\nReturn only the label."""
    try:
        raw = get_llm().invoke(prompt).content.strip().upper()
        label = raw.split()[0].strip(',.;!')
        if label not in {'SMALLTALK','EXPLORE','PLAN'}:
            return 'PLAN'
        return label
    except Exception:
        return 'PLAN'

def _extract_slots(user_input: str, context: Dict[str,Any]) -> Dict[str,Any]:
    updated = {}
    t = user_input.lower()
    # Travelers
    m = re.search(r'(?:for|we are|we\'re|with)?\s*(\d{1,2})\s*(?:traveler|travellers|travelers|people|persons|ppl|guys)', t)
    if m:
        try:
            num = int(m.group(1))
            if 0 < num < 20:
                updated['number_of_travelers'] = num
        except:  # noqa
            pass
    # Duration
    m = re.search(r'(\d{1,2})\s*(?:day|days)\b', t)
    if m:
        try:
            d = int(m.group(1))
            if 1 <= d <= 60:
                updated['duration_days'] = d
        except:  # noqa
            pass
    # ISO date
    m = re.search(r'(20\d{2}-\d{2}-\d{2})', user_input)
    if m and not context.get('committed_start_date'):
        updated['start_date'] = m.group(1)
    # Month mention
    months = {m.lower(): i+1 for i,m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'])}
    for name, idx in months.items():
        if re.search(rf'\b{name.lower()}\b', t) and not context.get('committed_start_date') and not context.get('start_date'):
            today = date.today(); year = today.year if idx >= today.month else today.year + 1
            updated['proposed_start_date'] = date(year, idx, 10).isoformat(); break
    # Origin city
    m = re.search(r'from\s+([A-Z][A-Za-z\s]{2,40})', user_input)
    if m and not context.get('user_city'):
        city = m.group(1).strip(); 
        if len(city.split()) <= 4: updated['user_city'] = city
    # Budget
    m = re.search(r'(?:USD|US\$|\$)\s?(\d{2,6})', user_input)
    if m: updated['budget'] = f"USD {m.group(1)}"
    # Destination selection
    if not context.get('destination'):
        m = re.search(r'(?:do|pick|choose|select|go to|let\'s do)\s+([A-Z][A-Za-z]{2,30})', user_input)
        if m: updated['destination'] = m.group(1)
    # Currency inference
    cur = re.search(r'\b(INR|USD|EUR|GBP|JPY|AUD|CAD)\b', user_input.upper())
    if cur and not context.get('user_currency'): updated['user_currency'] = cur.group(1)
    if context.get('committed_start_date') and 'start_date' in updated: updated.pop('start_date', None)
    return {k:v for k,v in updated.items() if k in ALLOWED_CONTEXT_KEYS}

def _next_missing_slot(context: Dict[str,Any]) -> str|None:
    for slot in SLOT_ORDER:
        if slot == 'start_date' and context.get('committed_start_date'): continue
        if not context.get(slot): return slot
    return None

def _init_slot_status(ctx: Dict[str,Any]):
    ss = ctx.get('slot_status') or {}
    for s in SLOT_ORDER:
        if s not in ss: ss[s] = 'unfilled'
        if ctx.get(s): ss[s] = 'committed'
    if ctx.get('committed_start_date'): ss['start_date'] = 'committed'
    ctx['slot_status'] = ss
    return ss

def _provenance_commit(ctx: Dict[str,Any], extracted: Dict[str,Any]):
    prov = ctx.get('_provenance', {})
    ts = time.time()
    for k,v in extracted.items():
        if k not in ctx or not ctx.get(k):
            ctx[k] = v
            prov[k] = {'value': v, 'source': 'user', 'ts': ts}
    ctx['_provenance'] = prov

def conversation_agent(state: ConversationAgentState):
    chat_history = state.get('chat_history', []) or []
    user_input = state.get('user_input','')
    ctx = state.get('context', {}) or {}
    history = _history_to_text(chat_history)
    intent = _classify_intent(user_input, history)
    state['planning_stage'] = 'plan' if intent == 'PLAN' else ('explore' if intent=='EXPLORE' else 'smalltalk')
    extracted = _extract_slots(user_input, ctx)
    _provenance_commit(ctx, extracted)
    # Accept proposed date
    if ctx.get('proposed_start_date') and any(t in user_input.lower() for t in ['use that','yes','ok','okay','sounds good','works','go with that']):
        ctx['committed_start_date'] = ctx['proposed_start_date']
        ctx['slot_status'] = ctx.get('slot_status', {})
        ctx['slot_status']['start_date'] = 'committed'
    # Auto commit ISO start_date
    if ctx.get('start_date') and re.match(r'^20\d{2}-\d{2}-\d{2}$', str(ctx['start_date'])):
        ctx['committed_start_date'] = ctx['start_date']
    _init_slot_status(ctx)
    missing = _next_missing_slot(ctx)
    if state['planning_stage'] == 'smalltalk':
        state['response'] = 'Tell me about the trip you have in mind or ask for ideas.'
        state['context'] = ctx; state['missing_info']=True if missing else False
        return state
    if state['planning_stage'] == 'explore':
        tool = get_destination_tool()
        try: ideas = tool.run(user_input or 'diverse travel preferences')
        except Exception: ideas = 'Here are some varied destinations to consider.'
        follow = 'Pick one to start planning.'
        state['response'] = ideas+"\n\n"+follow
        state['context'] = ctx; state['missing_info']=True
        return state
    if missing:
        prompts = {
            'user_city': 'What city will you depart from?',
            'destination': 'Where do you want to go?',
            'start_date': 'What start date? (YYYY-MM-DD or month name)',
            'duration_days': 'How many days?',
            'number_of_travelers': 'How many travelers?'
        }
        if missing=='start_date' and ctx.get('proposed_start_date'):
            state['response'] = f"Proposed date {ctx['proposed_start_date']}. Say 'use that' or give another." 
        else:
            state['response'] = prompts.get(missing, f"Provide {missing}.")
        ctx['slot_status'][missing] = 'asked'
        state['context']=ctx; state['missing_info']=True
        return state
    # Build or re-build plan (signature-based)
    core_sig = (
        ctx.get('user_city'),
        ctx.get('destination'),
        ctx.get('committed_start_date'),
        ctx.get('duration_days'),
        ctx.get('number_of_travelers')
    )
    prior_sig = state.get('plan_sig')
    if not state.get('plan') or prior_sig is None:
        state['plan'] = build_plan(ctx)
        state['tool_cursor']=0; state['tool_results']={}; state['errors']=[]
        state['plan_sig']=core_sig
    else:
        if core_sig != prior_sig:
            # core context changed -> full rebuild
            state['plan'] = build_plan(ctx)
            state['tool_cursor']=0; state['tool_results']={}; state['errors']=[]
            state['plan_sig']=core_sig
            state['response'] = 'Context changed, recalculating plan.'
        else:
            state['plan'] = replan(ctx, state['plan'])
    return planner_act(state)

async def _execute_step(step: PlanStep, ctx: Dict[str,Any]):
    name = step.step
    if name == 'flights':
        from tools.flight import get_flight_tool, format_flights_for_display
        tool = get_flight_tool()
        result = await run_tool('flights', tool.func, ctx.get('user_city'), ctx.get('destination'), ctx.get('committed_start_date'), ctx.get('number_of_travelers') or 1)
        if not result.ok:
            return False, {'error': result.error}
        validation = validate_flights(result.data)
        offers = validation['offers']
        if not offers:
            return False, {'error': 'no_valid_flights'}
        display = format_flights_for_display([o.model_dump() for o in offers])
        min_price = extract_min_price(offers)
        payload = {'raw': [o.model_dump() for o in offers], 'display': display, 'min_price': min_price, 'issues': validation['errors']}
        return True, payload
    if name == 'weather':
        from tools.weather import get_weather_tool
        tool = get_weather_tool()
        result = await run_tool('weather', tool.func, ctx.get('destination'), ctx.get('committed_start_date'))
        if not result.ok:
            return False, {'error': result.error}
        vw = validate_weather(result.data, ctx.get('committed_start_date'), ctx.get('destination'))
        return True, {'raw': result.data, 'snapshot': vw.get('snapshot').model_dump() if vw.get('snapshot') else None}
    if name == 'activities':
        from tools.activity import get_activity_tool
        tool = get_activity_tool(); result = await run_tool('activities', tool.func, ctx.get('destination'))
        return (result.ok, {'data': result.data} if result.ok else {'error': result.error})
    if name == 'nearby':
        from tools.map import get_map_tool
        tool = get_map_tool(); result = await run_tool('nearby', tool.func, ctx.get('destination'), 'tourism')
        return (result.ok, {'data': result.data} if result.ok else {'error': result.error})
    if name == 'budget':
        from tools.budget import get_budget_tool
        tool = get_budget_tool(); trip_details = {
            'destination': ctx.get('destination'),
            'flight_cost': ctx.get('flight_cost',''),
            'nights': max((ctx.get('duration_days') or 1)-1,1),
            'travelers': ctx.get('number_of_travelers') or 1,
            'activities': [],
            'days': ctx.get('duration_days') or 1
        }
        result = await run_tool('budget', tool.func, trip_details)
        return (result.ok, {'data': result.data} if result.ok else {'error': result.error})
    if name == 'assemble':
        from tools.assembler import get_assembler_tool
        tool = get_assembler_tool(); payload = {**ctx}
        result = await run_tool('assemble', tool.func, payload)
        return (result.ok, {'data': result.data} if result.ok else {'error': result.error})
    return False, {'error':'unknown_step'}

def _progress(plan: List[PlanStep]):
    done = [s.step for s in plan if s.status=='done']
    running = [s.step for s in plan if s.status=='running']
    pending = [s.step for s in plan if s.status in ('pending','error','skipped') and s.step not in running]
    return f"Progress | done: {done} running: {running} pending: {pending}".strip()

def planner_act(state: ConversationAgentState):
    ctx = state.get('context', {})
    plan: List[PlanStep] = state.get('plan', [])
    cursor = state.get('tool_cursor',0)
    if cursor >= len(plan):
        return planner_reflect(state)
    step = plan[cursor]
    step.status='running'
    async def run_current():
        ok, payload = await _execute_step(step, ctx)
        if ok:
            step.status='done'
            if step.step=='flights' and isinstance(payload.get('raw'), list):
                try:
                    prices=[f.get('price') for f in payload['raw'] if isinstance(f,dict) and 'price' in f and isinstance(f['price'],(int,float))]
                    if prices: ctx['flight_cost']=min(prices)
                except: pass
            if step.step=='weather' and payload.get('snapshot'):
                ctx['weather_snapshot']=payload['snapshot']
            state.setdefault('tool_results',{})[step.step]=payload
        else:
            step.status='error'; step.error = payload.get('error')
            state.setdefault('errors',[]).append(f"{step.step}:{step.error}")
        state['tool_cursor']=cursor+1
    asyncio.run(run_current())
    if state['tool_cursor'] < len(plan):
        state['response']=_progress(plan)
        state['context']=ctx
        return state
    return planner_reflect(state)

def planner_reflect(state: ConversationAgentState):
    plan: List[PlanStep] = state.get('plan', [])
    ctx = state.get('context', {})
    errors = state.get('errors', [])
    summary = []
    for s in plan:
        if s.status=='error': summary.append(f"{s.step} failed ({s.error})")
        elif s.status=='skipped': summary.append(f"{s.step} skipped")
    if not state.get('response') or state['response'].startswith('Progress'):
        state['response'] = 'Plan complete.'
    if summary:
        state['response'] += "\nIssues: " + "; ".join(summary)
    state['context']=ctx
    state['missing_info']=False
    return state

def context_extractor_node(state: TripPlannerState):
    return state

_GRAPH_CACHE = None

def build_conversation_graph():
    global _GRAPH_CACHE
    if _GRAPH_CACHE: return _GRAPH_CACHE
    graph = StateGraph(ConversationAgentState)
    graph.add_node('conversation', conversation_agent)
    graph.add_node('planner_act', planner_act)
    graph.add_node('planner_reflect', planner_reflect)
    def router(state):
        return 'planner_act' if state.get('plan') else END
    graph.add_conditional_edges('conversation', router, {'planner_act':'planner_act', END: END})
    def act_router(state):
        if state.get('plan') and state.get('tool_cursor',0) < len(state.get('plan')):
            return 'planner_act'
        return 'planner_reflect'
    graph.add_conditional_edges('planner_act', act_router, {'planner_act':'planner_act','planner_reflect':'planner_reflect'})
    graph.add_edge('planner_reflect', END)
    graph.set_entry_point('conversation')
    memory = MemorySaver(); app = graph.compile(checkpointer=memory)
    _GRAPH_CACHE = app
    return app

def run_conversation_graph(user_input, chat_history, context=None):
    context = context or {}
    state: ConversationAgentState = {'user_input': user_input,'chat_history': chat_history,'context': context}
    app = build_conversation_graph()
    return app.invoke(state)

    def _next_missing_slot(context: Dict[str,Any]) -> str|None:
        for slot in SLOT_ORDER:
            if slot == 'start_date' and context.get('committed_start_date'):
                continue
            if not context.get(slot):
                return slot
        return None

    def conversation_agent(state: ConversationAgentState):
        chat_history = state.get('chat_history', []) or []
        user_input = state.get('user_input', '')
        context = state.get('context', {}) or {}
        history_str = _history_to_text(chat_history)
        intent = _classify_intent(user_input, history_str)
        state['planning_stage'] = 'plan' if intent == 'PLAN' else ('explore' if intent == 'EXPLORE' else 'smalltalk')
        extracted = _extract_slots(user_input, context)
        for k,v in extracted.items():
            if k not in context or not context.get(k):
                context[k] = v
        if context.get('start_date') and re.match(r'^20\d{2}-\d{2}-\d{2}$', str(context['start_date'])):
            context['committed_start_date'] = context['start_date']
        if context.get('proposed_start_date') and any(t in user_input.lower() for t in ['use that','sounds good','ok','okay','yes','let\'s go with','works']):
            if not context.get('committed_start_date'):
                context['committed_start_date'] = context['proposed_start_date']
                context.pop('start_date', None)
        missing_slot = _next_missing_slot(context)
        if state['planning_stage'] == 'smalltalk':
            state['response'] = "Hi! Tell me about the kind of trip you're thinking about or ask for ideas." if not user_input.strip() else "Share any trip ideas or ask for destination suggestions when ready."
            state['context'] = context
            return state
        if state['planning_stage'] == 'explore':
            dest_tool = get_destination_tool()
            prefs = user_input if user_input.strip() else "beach, culture, moderate budget"
            try:
                suggestions = dest_tool.run(prefs)
            except Exception:
                suggestions = "Here are a few diverse destinations you might enjoy."
            follow = "Pick one destination to move into planning." if not context.get('destination') else f"Great, we can start planning for {context['destination']}. Provide a month or date to begin."
            state['response'] = suggestions + "\n\n" + follow
            state['context'] = context
            return state
        if missing_slot:
            prompts = {
                'user_city': "What city will you depart from?",
                'destination': "Where do you want to go? You can also ask for ideas first.",
                'start_date': "What is your start date? Give YYYY-MM-DD or just a month name.",
                'duration_days': "How many days should the trip last?",
                'number_of_travelers': "How many travelers are going?"
            }
            if missing_slot == 'start_date' and context.get('proposed_start_date'):
                state['response'] = f"We could start around {context['proposed_start_date']}. Say 'use that' to accept or give another date."
            else:
                state['response'] = prompts.get(missing_slot, f"Provide {missing_slot} please.")
            state['missing_info'] = True
            state['context'] = context
            return state
        state['missing_info'] = False
        state['context'] = context
        return trip_planner_node(state)

    def context_extractor_node(state: TripPlannerState):
        return state

    _GRAPH_CACHE = None

    def build_conversation_graph():
        global _GRAPH_CACHE
        if _GRAPH_CACHE:
            return _GRAPH_CACHE
        graph = StateGraph(ConversationAgentState)
        graph.add_node('conversation', conversation_agent)
        graph.add_edge('conversation', END)
        graph.set_entry_point('conversation')
        memory = MemorySaver()
        app = graph.compile(checkpointer=memory)
        _GRAPH_CACHE = app
        return app

    def run_conversation_graph(user_input, chat_history, context=None):
        context = context or {}
        state: ConversationAgentState = {
            'user_input': user_input,
            'chat_history': chat_history,
            'context': context
        }
        app = build_conversation_graph()
        result = app.invoke(state)
        return result
