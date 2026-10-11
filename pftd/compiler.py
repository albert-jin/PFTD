"""Fail-closed compiler; semantic specifications/oracles are inputs, never guessed.

All intervals are zero-based and half-open. A root without verified plans remains
available for full-task training. Independently checked parent plans are recompiled
from the complete program, never composed from child prompts.
"""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path
from pftd.checker import Checker
from pftd.codec import canonical, decode, encode
from pftd.contracts import equal, has_type, reject_answer_leakage, validate_contract
from pftd.interfaces import candidate_spans, interface, source
from pftd.sandbox import InfrastructureError


def validate_splits(roots):
    families, questions, ids = {}, {}, set()
    for root in roots:
        if root['root_id'] in ids:
            raise ValueError('duplicate root ID')
        ids.add(root['root_id'])
        if root['split'] not in {'train', 'dev', 'test'}:
            raise ValueError('invalid root split')
        normalized = re.sub(r'\s+', ' ', re.sub(r'\d+(?:\.\d+)?', '<number>', root['question'].lower())).strip()
        for mapping, key in [(families, root['family_id']), (questions, normalized)]:
            if key in mapping and mapping[key] != root['split']:
                raise ValueError('root/near-duplicate family crosses data splits')
            mapping[key] = root['split']


def wrapped_reference(program, span, inputs, outputs, public_inputs):
    body = source(ast.parse(program).body[span[0]:span[1]])
    rename = '\n'.join(f'{old} = {new}' for old, new in zip(inputs, public_inputs) if old != new)
    result = outputs[0] if len(outputs) == 1 else '(' + ', '.join(outputs) + ',)'
    indented = '\n'.join('    ' + line for line in (rename + '\n' + body + '\nreturn ' + result).strip().splitlines())
    return 'def solve_unit(' + ', '.join(public_inputs) + '):\n' + indented


def mutate(reference):
    """Concrete mutation operators; only oracle-disagreeing mutants are known-wrong."""
    tree = ast.parse(reference)
    results = []
    for index, node in enumerate(ast.walk(tree)):
        for operation in ('flip', 'drop', 'off_by_one'):
            clone = copy.deepcopy(tree)
            target = list(ast.walk(clone))[index]
            changed = False
            if operation == 'flip' and isinstance(target, ast.Compare):
                inverse = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.Gt: ast.LtE, ast.LtE: ast.Gt, ast.GtE: ast.Lt}
                if type(target.ops[0]) in inverse:
                    target.ops[0] = inverse[type(target.ops[0])]()
                    changed = True
            if operation == 'drop' and isinstance(target, ast.comprehension) and target.ifs:
                target.ifs.pop()
                changed = True
            if operation == 'drop' and isinstance(target, ast.BoolOp) and len(target.values) > 1:
                target.values.pop()
                changed = True
            if operation == 'off_by_one' and isinstance(target, ast.Call) and isinstance(target.func, ast.Name) and target.func.id == 'range' and target.args:
                pos = 0 if len(target.args) == 1 else 1
                target.args[pos] = ast.BinOp(left=target.args[pos], op=ast.Sub(), right=ast.Constant(1))
                changed = True
            if changed:
                results.append((operation, ast.unparse(ast.fix_missing_locations(clone))))
    clone = copy.deepcopy(tree)
    next(n for n in clone.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_unit').body = [ast.Return(value=ast.Constant(0))]
    results.append(('constant', ast.unparse(ast.fix_missing_locations(clone))))
    for index, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Return) and node.value:
            clone = copy.deepcopy(tree)
            target = list(ast.walk(clone))[index]
            target.value = ast.Subscript(value=target.value, slice=ast.Slice(upper=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))), ctx=ast.Load())
            results.append(('missing_element', ast.unparse(ast.fix_missing_locations(clone))))
    return list(dict.fromkeys(results))[:32]


class Compiler:
    def __init__(self, sandbox, contract_generator=None):
        self.sandbox, self.contract_generator = sandbox, contract_generator

    def compile_unit(self, root, span, children=()):
        if root['split'] != 'train' or root.get('teacher_correct') is not True:
            raise ValueError('only correct teacher trajectories of training roots may compile')
        program = root['program']
        nodes = ast.parse(program).body
        lo, hi = span
        live_in, live_out = interface(program, lo, hi)
        if not live_in or not live_out:
            raise ValueError('span has no functional interface')
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {'print', 'final_answer'} for node in nodes[lo:hi] for n in ast.walk(node)):
            raise ValueError('print/submission side effects are not unit outputs')
        parameter_map = root['parameter_map']
        if any(index >= lo for index in parameter_map.values()):
            raise ValueError('parameters must be explicit root inputs established before the span')
        key = f'{lo}:{hi}'
        plan = root.get('span_plans', {}).get(key)
        if plan is None or not plan.get('oracle_code') or not plan.get('oracle_provenance'):
            raise ValueError('missing independent semantic oracle for this span')
        if plan['oracle_code'].strip() == source(nodes[lo:hi]).strip():
            raise ValueError('teacher span cannot serve as its own independent oracle')
        contexts = root['reward_parameters']
        scheduled = root['readiness_parameters']
        if len(contexts) != 6 or len(scheduled) != 4:
            raise ValueError('paper protocol requires six reward and four readiness contexts')
        keys = [canonical(p) for p in contexts + scheduled]
        if len(set(keys)) != 10 or canonical(root['original_parameters']) not in keys[:6]:
            raise ValueError('contexts must be unique/disjoint and include the original reward instance')
        question_template = root['question_template']
        if question_template.format(**root['original_parameters']) != root['question']:
            raise ValueError('problem parameterization must reproduce the original question')
        records = []
        for parameters in contexts + scheduled:
            if set(parameters) != set(parameter_map):
                raise ValueError('parameter assignment does not match explicit root map')
            question = question_template.format(**parameters)
            # A trusted independent root oracle validates both domain and final answer.
            oracle_answer = self.sandbox.run({'mode': 'call', 'code': root['answer_oracle_code'], 'args': encode([parameters]), 'imports': root.get('imports', [])})
            if not oracle_answer['ok']:
                raise ValueError('invalid parameterization / independent answer oracle failed')
            request = {'mode': 'program', 'code': program, 'span': [lo, hi], 'capture': live_in, 'outputs': live_out,
                'parameter_map': parameter_map, 'parameters': encode(parameters), 'imports': root.get('imports', [])}
            a, b = self.sandbox.run(request), self.sandbox.run(request)
            if not a['ok'] or not b['ok'] or a != b:
                raise ValueError('teacher replay is unsupported or nondeterministic')
            if canonical(decode(a['answer'])) != canonical(decode(oracle_answer['value'])):
                raise ValueError('fixed teacher path disagrees with independent final-answer oracle')
            inputs = decode(a['entry'])
            oracle = self.sandbox.run({'mode': 'call', 'code': plan['oracle_code'], 'args': encode([inputs[k] for k in live_in]), 'imports': root.get('imports', [])})
            if not oracle['ok']:
                raise ValueError('independent unit oracle failed')
            records.append({'parameters': encode(parameters), 'question': question, 'inputs': a['entry'],
                'expected_output': a['exit'], 'expected_answer': a['answer'], 'oracle_output': oracle['value']})
        if len({canonical(decode(c['expected_output'])) for c in records}) < 2:
            raise ValueError('constant output across valid contexts')
        unit_id = hashlib.sha256((root['root_id'] + ':' + key).encode()).hexdigest()[:16]
        contract = plan.get('contract')
        if self.contract_generator is not None:
            contract = self.contract_generator(root['question'], source(nodes[lo:hi]), live_in, live_out, records[0]['inputs'])
        if contract is None:
            raise ValueError('missing teacher-generated public contract')
        public_inputs = list(contract['input_schema'])
        if len(public_inputs) != len(live_in):
            raise ValueError('renamed input arity mismatch')
        public = {'unit_id': unit_id, 'function_name': 'solve_unit', **copy.deepcopy(contract),
            'response_format': 'Thought and Code; defines solve_unit; no final_answer'}
        validate_contract(public, public_inputs)
        reject_answer_leakage(public, root['question'], [decode(c['expected_output']) for c in records] + [decode(c['expected_answer']) for c in records])
        reference = wrapped_reference(program, span, live_in, live_out, public_inputs)
        # Re-import only declared libraries; no prefix state is copied into this call.
        import_nodes = [n for n in nodes[:hi] if isinstance(n, (ast.Import, ast.ImportFrom))]
        reference = source(import_nodes) + '\n' + reference
        for record in records:
            raw = decode(record['inputs'])
            renamed = dict(zip(public_inputs, [raw[k] for k in live_in]))
            if any(not has_type(v, public['input_schema'][k]['type']) for k, v in renamed.items()):
                raise ValueError('input type contradicts contract')
            record['inputs'] = encode(renamed)
            expected, independent = decode(record['expected_output']), decode(record.pop('oracle_output'))
            if not equal(expected, independent, public['output_schema']):
                raise ValueError('reference output not independently verified under contract semantics')
            dynamic = self.sandbox.run({'mode': 'call', 'code': reference, 'args': encode(list(renamed.values())), 'imports': public['allowed_imports']})
            if not dynamic['ok'] or not equal(decode(dynamic['value']), expected, public['output_schema']):
                raise ValueError('dependency closure failed in a fresh input-only process')
        private = {'root_id': root['root_id'], 'family_id': root['family_id'], 'split': 'train',
            'task_type': root.get('task_type', 'math'),
            'teacher_trace_hash': hashlib.sha256(program.encode()).hexdigest(), 'span': [lo, hi],
            'live_in': live_in, 'live_out': live_out, 'program': program, 'reference_function': reference,
            'parameter_map': parameter_map, 'imports': root.get('imports', []),
            'contexts': records[:6], 'schedule_contexts': records[6:],
            'verification_level': 'semantic_checked', 'oracle_provenance': plan['oracle_provenance'],
            'child_ids': list(children), 'mutant_checks': []}
        checker = Checker(self.sandbox, workers=1)
        for operator, mutant in mutate(reference) + [('supplied', x) for x in plan.get('mutants', [])]:
            # Establish a semantic counterexample first; syntax errors are not evidence.
            wrong = False
            for ctx in records:
                result = self.sandbox.run({'mode': 'call', 'code': mutant, 'args': encode(list(decode(ctx['inputs']).values())), 'imports': public['allowed_imports']})
                if result['ok'] and not equal(decode(result['value']), decode(ctx['expected_output']), public['output_schema']):
                    wrong = True
                    break
            if not wrong:
                continue
            reward = checker.check('Code:\n```py\n' + mutant + '\n```<end_code>', public, private)
            if reward == 1.0:
                raise ValueError('known-wrong mutant passes all reward contexts')
            private['mutant_checks'].append({'operator': operator, 'reward': reward})
        if not private['mutant_checks']:
            raise ValueError('no independently exposed wrong mutant')
        return {'public': public, 'private': private}

    def compile_roots(self, roots):
        validate_splits(roots)
        units, rejected = [], []
        for root in roots:
            if root['split'] != 'train' or not root.get('teacher_correct'):
                continue
            signatures, count, occupied = set(), 0, set()
            for span in candidate_spans(root['program'], root.get('turn_ends', [])):
                try:
                    if occupied.intersection(range(*span)):
                        continue
                    unit = self.compile_unit(root, span)
                    sig = ast.dump(ast.parse(unit['private']['reference_function']), include_attributes=False)
                    if sig in signatures:
                        continue
                    signatures.add(sig)
                    units.append(unit)
                    occupied.update(range(*span))
                    count += 1
                    if count == 3:
                        break
                except (ValueError, KeyError, SyntaxError, TypeError) as exc:
                    rejected.append({'root_id': root['root_id'], 'span': span, 'reason': str(exc)})
        return units, rejected

    def merge(self, root, first, second):
        a, b = first['private'], second['private']
        if a['root_id'] != b['root_id'] or a['span'][1] != b['span'][0]:
            raise ValueError('only adjacent spans of one root may merge')
        return self.compile_unit(root, [a['span'][0], b['span'][1]], [first['public']['unit_id'], second['public']['unit_id']])


def save_units(units, destination):
    dest = Path(destination)
    (dest / 'public').mkdir(parents=True, exist_ok=True)
    (dest / 'private').mkdir(parents=True, exist_ok=True)
    for unit in units:
        uid = unit['public']['unit_id']
        for view in ('public', 'private'):
            path = dest / view / (uid + '.json')
            path.write_text(json.dumps(unit[view], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            if view == 'private':
                path.chmod(0o600)
