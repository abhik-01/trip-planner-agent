from __future__ import annotations
from typing import List, Dict, Any
from utils.models import PlanStep

CORE_ORDER = ["flights","weather"]
ENRICH_ORDER = ["activities","nearby","budget","assemble"]

CRITICAL = {"flights": False, "weather": False}


def build_plan(ctx: Dict[str,Any]) -> List[PlanStep]:
    steps: List[PlanStep] = []
    have_origin = bool(ctx.get('user_city'))
    have_dest = bool(ctx.get('destination'))
    have_date = bool(ctx.get('committed_start_date'))

    # Flights only if we have origin+dest+date
    if have_origin and have_dest and have_date:
        steps.append(PlanStep(id="1", step="flights", requires=['user_city','destination','committed_start_date']))
    # Weather if destination + date
    if have_dest and have_date:
        steps.append(PlanStep(id="2", step="weather", requires=['destination','committed_start_date']))

    # Enrichment independent of flights
    if have_dest:
        steps.append(PlanStep(id="3", step="activities", requires=['destination']))
        steps.append(PlanStep(id="4", step="nearby", requires=['destination']))

    # Budget after we know duration & destination (flight cost helpful but optional)
    if have_dest and ctx.get('duration_days'):
        steps.append(PlanStep(id="5", step="budget", requires=['destination']))

    # Assemble always last
    steps.append(PlanStep(id="6", step="assemble", requires=[]))

    # Re-index ids sequentially
    for i,s in enumerate(steps, start=1):
        s.id = str(i)
    return steps


def replan(ctx: Dict[str,Any], previous: List[PlanStep]) -> List[PlanStep]:
    # If destination, date, or origin changed -> rebuild from scratch
    key_hash = (ctx.get('user_city'), ctx.get('destination'), ctx.get('committed_start_date'))
    prev_keys = getattr(replan, '_prev_keys', None)
    if prev_keys != key_hash:
        new_plan = build_plan(ctx)
        replan._prev_keys = key_hash  # type: ignore
        return new_plan
    # Otherwise keep previous plan (could add adaptive pruning here)
    return previous
