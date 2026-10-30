"""Strict JSON diagnostics: normalize numerical metadata, never stringify it."""
from pathlib import Path
import numpy as np
from ..runtime import atomic_json as _atomic_json


def json_value(value):
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key:json_value(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    # Runtime still rejects NaN/Infinity and unknown objects. Conversion happens
    # before replacement, so failed serialization cannot destroy a valid report.
    _atomic_json(path,json_value(value))
