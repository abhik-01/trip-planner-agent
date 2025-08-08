from __future__ import annotations
from typing import Callable, Any, Dict, List
import time, asyncio
from utils.models import ToolResult, FlightOffer, WeatherSnapshot
from typing import Optional

ERROR_MAP = {
    'network': 'network_error',
    'timeout': 'network_error',
    'invalid': 'invalid_input',
}

async def _maybe_await(value):
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value

def classify_error(e: Exception) -> str:
    txt = str(e).lower()
    if 'timeout' in txt or 'connection' in txt or 'network' in txt:
        return 'network_error'
    if 'invalid' in txt or 'valueerror' in txt:
        return 'invalid_input'
    return 'external_failure'

async def run_tool(name: str, fn: Callable, *args, **kwargs) -> ToolResult:
    start = time.time()
    try:
        raw = fn(*args, **kwargs)
        raw = await _maybe_await(raw)
        return ToolResult(ok=True, data=raw, meta={'tool': name, 'latency_ms': int((time.time()-start)*1000)})
    except Exception as e:
        return ToolResult(ok=False, error=f"{classify_error(e)}: {e}", meta={'tool': name, 'latency_ms': int((time.time()-start)*1000)})

async def run_parallel(batch: Dict[str, tuple]) -> Dict[str, ToolResult]:
    # batch: key -> (fn, args, kwargs)
    tasks = {k: asyncio.create_task(run_tool(k, fn, *a, **kw)) for k,(fn,a,kw) in batch.items()}
    results: Dict[str, ToolResult] = {}
    for k,t in tasks.items():
        try:
            results[k] = await t
        except Exception as e:
            results[k] = ToolResult(ok=False, error=f"external_failure: {e}")
    return results

# ---------------- Validation Helpers ----------------
def validate_flights(raw: Any) -> Dict[str, Any]:
    """Return dict with 'offers' (list[FlightOffer]) and optional 'errors'."""
    out = {'offers': [], 'errors': []}
    if not isinstance(raw, list):
        out['errors'].append('not_list')
        return out
    for item in raw:
        if not isinstance(item, dict):
            out['errors'].append('non_dict_item'); continue
        if 'error' in item:
            out['errors'].append(item.get('error') or 'flight_error'); continue
        try:
            offer = FlightOffer(
                price=float(item['price']),
                currency=str(item.get('currency','USD')),
                departure_airport=str(item['departure_airport']),
                arrival_airport=str(item['arrival_airport']),
                departure_time=str(item['departure_time']),
                airline=str(item.get('airline',''))
            )
            out['offers'].append(offer)
        except Exception as e:
            out['errors'].append(f'bad_item:{e}')
    return out

def validate_weather(raw: Any, date: Optional[str]=None, city: Optional[str]=None) -> Dict[str, Any]:
    if not isinstance(raw, str):
        return {'snapshot': None, 'error': 'not_string'}
    summary = raw.strip()
    snap = WeatherSnapshot(date=date or '', summary=summary)
    return {'snapshot': snap}

def extract_min_price(offers: List[FlightOffer]) -> Optional[float]:
    if not offers:
        return None
    try:
        return min(o.price for o in offers if isinstance(o.price, (int,float)))
    except Exception:
        return None
