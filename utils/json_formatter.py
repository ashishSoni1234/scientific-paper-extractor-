"""
json_formatter.py — Clean and validate the final JSON output.

Ensures the aggregated result dict can be safely serialised to JSON,
handles non-serialisable types, and produces a formatted JSON string
ready for download.
"""

import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def format_output_json(result: dict) -> str:
    """
    Serialise the result dict to a clean, indented JSON string.

    Handles edge cases like NaN/Inf floats, numpy arrays, and other
    non-standard Python types that would break standard json.dumps().

    Args:
        result: The aggregated result dict from aggregator.aggregate_results().

    Returns:
        Pretty-printed JSON string, validated with json.loads() before return.

    Raises:
        ValueError: If the result cannot be serialised even after sanitisation.
    """
    sanitised = _sanitise_for_json(result)

    try:
        json_string = json.dumps(sanitised, indent=2, ensure_ascii=False)
        # Verify the output is valid JSON
        json.loads(json_string)
        return json_string
    except (TypeError, ValueError) as exc:
        logger.error("JSON serialisation failed: %s", exc)
        raise ValueError(f"Could not serialise result to JSON: {exc}") from exc


def _sanitise_for_json(obj: Any) -> Any:
    """
    Recursively sanitise an object for JSON serialisation.

    Handles:
    - numpy arrays → lists
    - numpy scalars → native Python types
    - NaN/Inf floats → null / None
    - datetime objects → ISO strings
    - Everything else → str() as last resort

    Args:
        obj: Any Python object.

    Returns:
        JSON-serialisable version of the object.
    """
    if obj is None:
        return None

    # Handle numpy types if available
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            value = float(obj)
            return None if (math.isnan(value) or math.isinf(value)) else value
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {str(key): _sanitise_for_json(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_sanitise_for_json(item) for item in obj]

    if isinstance(obj, (int, float, bool, str)):
        return obj

    # Handle datetime
    try:
        return obj.isoformat()
    except AttributeError:
        pass

    # Last resort
    return str(obj)
