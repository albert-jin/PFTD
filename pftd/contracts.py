"""Public prompts, structural type checking and immutable comparison semantics."""
import ast
import json
import math
import re
from fractions import Fraction

ALLOWED_IMPORTS = {"math", "fractions", "itertools", "functools", "collections", "numpy", "sympy"}
PUBLIC_FIELDS = {"unit_id", "function_name", "task_spec", "input_schema", "output_schema", "allowed_imports", "response_format"}
TEACHER_FIELDS = {"task_spec", "input_schema", "output_schema", "allowed_imports"}


def validate_contract(record, inputs):
    if set(record) != PUBLIC_FIELDS or record["function_name"] != "solve_unit":
        raise ValueError("public record must have exactly the whitelisted fields")
    if list(record["input_schema"]) != list(inputs):
        raise ValueError("input schema must match the ordered, renamed live-in interface")
    if not isinstance(record["task_spec"], str) or not record["task_spec"].strip():
        raise ValueError("empty specification")
    if not isinstance(record["allowed_imports"], list) or not set(record["allowed_imports"]) <= ALLOWED_IMPORTS:
        raise ValueError("unsupported imports")
    for name, schema in record["input_schema"].items():
        if not name.isidentifier() or name.startswith("_") or not isinstance(schema.get("constraints", []), list):
            raise ValueError("invalid input schema")
        parse_type(schema["type"])
    parse_type(record["output_schema"]["type"])
    for key in ("order_matters", "duplicates_allowed"):
        if key in record["output_schema"] and type(record["output_schema"][key]) is not bool:
            raise ValueError("comparison flags must be boolean")
    for key, default in (("rtol", 1e-9), ("atol", 1e-12)):
        value = record["output_schema"].get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid fixed tolerance")


def render_prompt(record):
    # Deliberately select fields: private sentinels cannot change a prompt.
    public = {k: record[k] for k in PUBLIC_FIELDS}
    args = ", ".join(public["input_schema"])
    return (
        "You are an expert assistant who writes reusable Python functions.\n"
        "Write one function that satisfies the specification below. Use the format:\n"
        "Thought: <your reasoning>\nCode:\n```py\n<a complete definition of the function>\n```<end_code>\n\n"
        f"Function name: solve_unit({args})\n"
        f"Input: {json.dumps(public['input_schema'], ensure_ascii=False)}\n"
        f"Requirement: {public['task_spec']}\n"
        f"Output: {json.dumps(public['output_schema'], ensure_ascii=False)}\n"
        f"Allowed imports: {', '.join(public['allowed_imports'])}\n\n"
        "Do not call final_answer and do not print test cases. The checker will call solve_unit directly."
    )


def parse_type(text):
    node = ast.parse(text, mode="eval").body
    def valid(n):
        if isinstance(n, ast.Name):
            return n.id in {"int", "float", "bool", "str", "Fraction", "list", "tuple", "set", "dict"}
        return isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in {"list", "tuple", "set", "dict"} and all(valid(x) for x in (n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]))
    if not valid(node):
        raise ValueError("unsupported type schema: " + text)
    return node


def has_type(value, schema):
    n = parse_type(schema) if isinstance(schema, str) else schema
    classes = {"int": int, "float": float, "bool": bool, "str": str, "Fraction": Fraction, "list": list, "tuple": tuple, "set": set, "dict": dict}
    if isinstance(n, ast.Name):
        return type(value) is classes[n.id]
    if type(value) is not classes[n.value.id]:
        return False
    args = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
    if n.value.id == "dict":
        return len(args) == 2 and all(has_type(k, args[0]) and has_type(v, args[1]) for k, v in value.items())
    if n.value.id == "tuple":
        return len(value) == len(args) and all(has_type(x, t) for x, t in zip(value, args))
    return len(args) == 1 and all(has_type(x, args[0]) for x in value)


def equal(actual, expected, schema):
    if not has_type(actual, schema["type"]) or not has_type(expected, schema["type"]):
        return False
    def same(a, b):
        if type(a) is not type(b):
            return False
        if type(a) is float:
            return math.isfinite(a) and math.isclose(a, b, rel_tol=schema.get("rtol", 1e-9), abs_tol=schema.get("atol", 1e-12))
        if type(a) in (tuple, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if type(a) is dict:
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        return a == b
    if type(actual) in (list, tuple):
        if not schema.get("duplicates_allowed", True) and any(same(x, y) for i, x in enumerate(actual) for y in actual[:i]):
            return False
        if not schema.get("order_matters", True):
            remaining = list(expected)
            for x in actual:
                index = next((i for i, y in enumerate(remaining) if same(x, y)), None)
                if index is None:
                    return False
                remaining.pop(index)
            return not remaining
    return same(actual, expected)


def reject_answer_leakage(contract, question, hidden_values):
    public_text = json.dumps(contract, ensure_ascii=False)
    numeric = lambda s: set(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", str(s)))
    # Schema tolerances are comparison rules, not computed answers.
    semantic = {k: v for k, v in contract.items() if k in ("task_spec", "input_schema")}
    if numeric(json.dumps(semantic)) & (numeric(hidden_values) - numeric(question)):
        raise ValueError("computed number leaked into contract")
    if re.search(r"(?:answer|result)_\d+", public_text, re.I):
        raise ValueError("result-bearing variable name")
