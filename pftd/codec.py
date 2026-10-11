"""Lossless, bounded JSON wire format. Never deserialize pickle from a candidate."""
import json
import math
from fractions import Fraction


def encode(value, depth=0):
    if depth > 40:
        raise ValueError("value nesting limit")
    enc = lambda x: encode(x, depth + 1)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite value")
        return value
    if isinstance(value, Fraction):
        return {"$fraction": [value.numerator, value.denominator]}
    if type(value) is list:
        return [enc(x) for x in value]
    if type(value) is tuple:
        return {"$tuple": [enc(x) for x in value]}
    if type(value) is set:
        return {"$set": sorted((enc(x) for x in value), key=lambda x: json.dumps(x, sort_keys=True))}
    if type(value) is dict:
        return {"$dict": [[enc(k), enc(v)] for k, v in value.items()]}
    raise TypeError("unsupported boundary value: " + type(value).__name__)


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {"$fraction"}:
            return Fraction(*value["$fraction"])
        if set(value) == {"$tuple"}:
            return tuple(map(decode, value["$tuple"]))
        if set(value) == {"$set"}:
            return set(map(decode, value["$set"]))
        if set(value) == {"$dict"}:
            return {decode(k): decode(v) for k, v in value["$dict"]}
        return {k: decode(v) for k, v in value.items()}
    return value


def canonical(value):
    return json.dumps(encode(value), sort_keys=True, ensure_ascii=False, allow_nan=False)
