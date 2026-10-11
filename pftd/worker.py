"""Isolated worker. Production execution requires the Docker boundary, not AST alone."""
import ast
import builtins
import contextlib
import importlib
import io
import json
import signal
import sys
from pftd.codec import decode, encode

SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs all any bool dict enumerate filter float frozenset int isinstance len list map max min next pow range reversed round set sorted str sum tuple zip Exception ValueError TypeError ZeroDivisionError AssertionError StopIteration"
).split()}
FORBIDDEN = {"open", "eval", "exec", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "help", "dir", "type", "memoryview"}
FORBIDDEN_ATTRS = {"random", "load", "save", "loadtxt", "savetxt", "fromfile", "tofile", "memmap", "ctypes", "f2py", "test", "bench", "system", "popen", "lambdify", "parse_expr", "sympify", "utilities"}


def validate(code, imports, candidate=False):
    tree = ast.parse(code)
    if len(code) > 100_000:
        raise ValueError("code limit")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.With, ast.Delete)):
            raise ValueError("unsupported syntax")
        if isinstance(node, ast.Name) and (node.id.startswith('_') or node.id in FORBIDDEN or (candidate and node.id == 'final_answer')):
            raise ValueError("forbidden name")
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in FORBIDDEN_ATTRS):
            raise ValueError("forbidden attribute")
        if isinstance(node, ast.Import):
            if any(a.name not in imports or (a.asname and a.asname.startswith('_')) for a in node.names):
                raise ValueError("import not allowed")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in imports or any(a.name == '*' or a.name.startswith('_') or a.name in FORBIDDEN_ATTRS for a in node.names):
                raise ValueError("import not allowed")
    return tree


def execute(payload):
    allowed = payload.get("imports", [])
    from pftd.contracts import ALLOWED_IMPORTS
    if not set(allowed) <= ALLOWED_IMPORTS:
        raise ValueError("unsupported library")
    captured = io.StringIO()
    answer = []
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in allowed:
            raise ValueError("import not allowed")
        return importlib.import_module(name)
    def final_answer(value):
        answer.append(value)
        return value
    env = {"__builtins__": dict(SAFE_BUILTINS, __import__=safe_import, print=lambda *args, **kw: print(*args, file=captured, **kw)), "final_answer": final_answer}
    # Libraries are loaded before the 2-second execution timer.
    for lib in allowed:
        importlib.import_module(lib)
    def expired(signum, frame):
        raise TimeoutError("execution time limit")
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, payload.get("seconds", 2))
    mode = payload["mode"]
    if mode == "call":
        tree = validate(payload["code"], allowed, candidate=True)
        if any(not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for n in tree.body):
            raise ValueError("candidate must only define functions and imports")
        with contextlib.redirect_stdout(captured):
            exec(compile(tree, '<candidate>', 'exec'), env)
            result = env['solve_unit'](*decode(payload['args']))
        return {"value": encode(result)}
    if mode not in {"program", "downstream"}:
        raise ValueError("unknown worker mode")
    nodes = validate(payload['code'], allowed).body
    params = decode(payload.get('parameters', {}))
    # Only explicitly mapped top-level literal assignments can be parameterized.
    for name, index in payload.get('parameter_map', {}).items():
        node = nodes[index]
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != name:
            raise ValueError("parameter map is not a literal assignment")
        ast.literal_eval(node.value)
        if name not in params:
            raise ValueError("missing parameter")
        node.value = ast.parse(repr(params[name]), mode='eval').body
    states = {}
    lo, hi = payload.get('span', [0, len(nodes)])
    capture = payload.get('capture', [])
    with contextlib.redirect_stdout(captured):
        for index, node in enumerate(nodes):
            if index == lo:
                inputs = {k: env[k] for k in capture}
                # Shared mutable state cannot cross a compiled boundary.
                seen = set()
                def no_alias(v):
                    if type(v) in (list, dict, set):
                        if id(v) in seen:
                            raise ValueError("aliased boundary state")
                        seen.add(id(v))
                    if type(v) in (list, tuple, set):
                        for item in v:
                            no_alias(item)
                    elif type(v) is dict:
                        for item in v.values():
                            no_alias(item)
                for v in inputs.values():
                    no_alias(v)
                states['entry'] = encode(inputs)
                if mode == 'downstream':
                    outputs = payload['outputs']
                    value = decode(payload['value'])
                    values = [value] if len(outputs) == 1 else value
                    if len(values) != len(outputs):
                        raise ValueError("boundary arity mismatch")
                    env.update(zip(outputs, values))
            if mode == 'downstream' and lo <= index < hi:
                continue
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<program>', 'exec'), env)
            if index + 1 == hi and mode == 'program':
                values = [env[k] for k in payload.get('outputs', [])]
                states['exit'] = encode(values[0] if len(values) == 1 else tuple(values))
    states.update(answer=encode(answer[-1] if answer else None), observations=captured.getvalue()[:100_000])
    return states


def main():
    try:
        payload = json.loads(sys.stdin.read(2_000_000))
        result = {"ok": True, **execute(payload)}
    except BaseException as exc:
        result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:1000]}
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.setitimer(signal.ITIMER_REAL, 0)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(encoded) > 2_000_000:
        encoded = json.dumps({"ok": False, "error_type": "OutputLimit", "error": "output limit"})
    print(encoded)


if __name__ == '__main__':
    main()
