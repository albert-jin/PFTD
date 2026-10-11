import re
from concurrent.futures import ThreadPoolExecutor
from pftd.codec import canonical, decode, encode
from pftd.contracts import equal
from pftd.sandbox import InfrastructureError


def extract_code(response):
    matches = re.findall(r'```(?:python|py)?\s*\n(.*?)```', response, re.S)
    if not matches:
        raise ValueError('missing CodeAct code block')
    return '\n'.join(matches)


class Checker:
    def __init__(self, sandbox, workers=32, mode='joint', answer_grader=None):
        if mode not in {'joint', 'local', 'downstream'}:
            raise ValueError('unknown reward mode')
        self.sandbox, self.workers, self.mode = sandbox, workers, mode
        self.answer_grader = answer_grader

    def context(self, code, public, private, context):
        args = decode(context['inputs'])
        candidate = self.sandbox.run({'mode': 'call', 'code': code, 'args': encode([args[n] for n in public['input_schema']]), 'imports': public['allowed_imports']})
        if not candidate['ok']:
            return False
        value, expected = decode(candidate['value']), decode(context['expected_output'])
        local = equal(value, expected, public['output_schema'])
        if self.mode != 'downstream' and not local:
            return False
        if self.mode == 'local':
            return local
        downstream = self.sandbox.run({'mode': 'downstream', 'code': private['program'],
            'span': private['span'], 'capture': private['live_in'], 'outputs': private['live_out'],
            'parameter_map': private['parameter_map'], 'parameters': context['parameters'],
            'value': candidate['value'], 'imports': private['imports']})
        # References were independently verified on admission; downstream errors induced
        # by candidate values are candidate failures, not infrastructure failures.
        if not downstream['ok']:
            return False
        if self.answer_grader:
            root = {'question': context['question'], 'answer': decode(context['expected_answer']), 'task_type': private.get('task_type', 'math')}
            try:
                return bool(self.answer_grader(root, decode(downstream['answer'])))
            except Exception as exc:
                raise InfrastructureError('downstream final-answer grader failed') from exc
        return canonical(decode(downstream['answer'])) == canonical(decode(context['expected_answer']))

    def check(self, response, public, private, readiness=False):
        if private['verification_level'] != 'semantic_checked':
            raise InfrastructureError('reference-only units must never receive rewards')
        contexts = private['schedule_contexts' if readiness else 'contexts']
        try:
            code = extract_code(response)
        except ValueError:
            return 0.0
        with ThreadPoolExecutor(max_workers=min(self.workers, len(contexts))) as pool:
            outcomes = list(pool.map(lambda ctx: self.context(code, public, private, ctx), contexts))
        return sum(outcomes) / len(outcomes)

    def group(self, responses, public, private, readiness=False):
        # Verify reference assets before each group. Corruption cannot become a negative reward.
        try:
            contexts = private['schedule_contexts' if readiness else 'contexts']
            for ctx in contexts:
                replay = self.sandbox.run({'mode': 'program', 'code': private['program'],
                    'span': private['span'], 'capture': private['live_in'], 'outputs': private['live_out'],
                    'parameter_map': private['parameter_map'], 'parameters': ctx['parameters'], 'imports': private['imports']})
                if not replay['ok'] or canonical(decode(replay['exit'])) != canonical(decode(ctx['expected_output'])) or canonical(decode(replay['answer'])) != canonical(decode(ctx['expected_answer'])):
                    raise InfrastructureError('reference replay mismatch')
            return [self.check(r, public, private, readiness) for r in responses], None
        except InfrastructureError as exc:
            return None, str(exc)
