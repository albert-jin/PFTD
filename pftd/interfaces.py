"""Conservative statement-level may-read/may-write/must-write analysis."""
import ast
import builtins
from dataclasses import dataclass


class Unsupported(ValueError):
    pass


@dataclass
class Flow:
    reads: set
    writes: set
    must: set


def names(node, context=ast.Load):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, context)}


def expression_reads(node):
    if node is None:
        return set()
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        reads, bound = set(), set()
        for g in node.generators:
            reads |= expression_reads(g.iter) - bound
            bound |= names(g.target, ast.Store)
            for cond in g.ifs:
                reads |= expression_reads(cond) - bound
        for child in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
            reads |= expression_reads(child) - bound
        return reads
    if isinstance(node, ast.Lambda):
        return expression_reads(node.body) - {a.arg for a in node.args.args}
    if isinstance(node, ast.Name):
        return {node.id} if isinstance(node.ctx, ast.Load) else set()
    reads = set()
    for child in ast.iter_child_nodes(node):
        reads |= expression_reads(child)
    return reads


def block(nodes):
    reads, writes, must = set(), set(), set()
    for node in nodes:
        f = statement(node)
        reads |= f.reads - must
        writes |= f.writes
        must |= f.must
    return Flow(reads, writes, must)


def statement(n):
    if isinstance(n, ast.If):
        a, b = block(n.body), block(n.orelse)
        return Flow(expression_reads(n.test) | a.reads | b.reads, a.writes | b.writes, a.must & b.must)
    if isinstance(n, (ast.For, ast.While)):
        body, other = block(n.body), block(n.orelse)
        target = names(n.target, ast.Store) if isinstance(n, ast.For) else set()
        read = expression_reads(n.iter if isinstance(n, ast.For) else n.test)
        return Flow(read | (body.reads - target) | other.reads, target | body.writes | other.writes, set())
    if isinstance(n, ast.Try):
        # Conservatively accept branch reads, but no conditional write kills liveness.
        parts = [block(n.body), block(n.orelse), block(n.finalbody)] + [block(h.body) for h in n.handlers]
        return Flow(set().union(*(p.reads for p in parts)), set().union(*(p.writes for p in parts)), set())
    if isinstance(n, ast.FunctionDef):
        if n.decorator_list:
            raise Unsupported("decorated teacher functions require explicit effect analysis")
        args = {a.arg for a in [*n.args.posonlyargs, *n.args.args, *n.args.kwonlyargs]}
        args |= {a.arg for a in [n.args.vararg, n.args.kwarg] if a is not None}
        body = block(n.body)
        defaults = set().union(*(expression_reads(x) for x in [*n.args.defaults, *n.args.kw_defaults]))
        # Free names are conservatively treated as reads at definition time.
        return Flow((body.reads - args - {n.name}) | defaults, {n.name}, {n.name})
    if isinstance(n, (ast.AsyncFunctionDef, ast.ClassDef)):
        raise Unsupported("async/class definitions are outside the supported value domain")
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        imports = {a.asname or (a.name.split('.')[0] if isinstance(n, ast.Import) else a.name) for a in n.names}
        return Flow(set(), imports, imports)
    reads = expression_reads(n)
    writes = names(n, ast.Store)
    if isinstance(n, ast.AugAssign):
        reads |= names(n.target, ast.Store)
    for child in ast.walk(n):
        if isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(child.ctx, ast.Store):
            base = child.value
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if isinstance(base, ast.Name):
                reads.add(base.id)
                writes.add(base.id)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr in {"append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse", "update", "add", "discard"}:
            if isinstance(child.func.value, ast.Name):
                reads.add(child.func.value.id)
                writes.add(child.func.value.id)
    return Flow(reads, writes, writes)


def interface(program, lo, hi):
    """Zero-based half-open spans; loops/branches remain atomic top-level statements."""
    nodes = ast.parse(program).body
    if not 0 <= lo < hi <= len(nodes):
        raise ValueError("invalid statement span")
    span, tail = block(nodes[lo:hi]), block(nodes[hi:])
    imported = set()
    for n in nodes[:hi]:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imported |= statement(n).writes
    external = set(dir(builtins)) | {"final_answer"} | imported
    return sorted(span.reads - external), sorted(span.writes & tail.reads - imported)


def candidate_spans(program, turn_ends=()):
    n = len(ast.parse(program).body)
    bounds = sorted({0, n, *turn_ends})
    proposed = [(a, b) for a, b in zip(bounds, bounds[1:])]
    proposed += [(i, i + 1) for i in range(n)]
    return list(dict.fromkeys(proposed))[:12]


def source(nodes):
    return '\n'.join(ast.unparse(n) for n in nodes)
