"""Restricted contract requests. Only caller-approved local/remote clients are used."""
import json
from pftd.codec import decode
from pftd.contracts import ALLOWED_IMPORTS, TEACHER_FIELDS, validate_contract


def type_name(value):
    from fractions import Fraction
    if isinstance(value, Fraction):
        return 'Fraction'
    if type(value) in (int, float, bool, str):
        return type(value).__name__
    if type(value) in (list, set):
        types = {type_name(v) for v in value}
        if len(types) != 1:
            raise ValueError('supply an explicit homogeneous schema for empty/mixed containers')
        return type(value).__name__ + '[' + next(iter(types)) + ']'
    if type(value) is tuple:
        return 'tuple[' + ', '.join(map(type_name, value)) + ']'
    if type(value) is dict:
        keys, vals = {type_name(k) for k in value}, {type_name(v) for v in value.values()}
        if len(keys) != 1 or len(vals) != 1:
            raise ValueError('explicit dictionary schema needed')
        return 'dict[' + next(iter(keys)) + ', ' + next(iter(vals)) + ']'
    raise ValueError('unsupported interface type')


class TeacherContractGenerator:
    def __init__(self, client, model='Qwen/Qwen2.5-32B-Instruct'):
        self.client, self.model = client, model
        self.teacher_tokens = 0

    def __call__(self, question, span_code, live_in, live_out, entry):
        # Only types are sent; entry values, outputs, suffix and answers are absent.
        values = decode(entry)
        types = {name: type_name(values[name]) for name in live_in}
        prompt = (
            'Describe the FUNCTION computed by the following Python fragment so another programmer can implement it.\n'
            f'Original problem: {question}\nFragment:\n{span_code}\n'
            f'Input names/types: {json.dumps(types)}\nOutput names: {json.dumps(live_out)}\n'
            'Return only a JSON object with exactly task_spec, input_schema, output_schema, allowed_imports. '
            'task_spec states WHAT, not HOW. Input schema maps each input name to type and constraints. '
            'Output schema declares type, order_matters, duplicates_allowed and optional rtol/atol. '
            f'Allowed modules: {sorted(ALLOWED_IMPORTS)}. '
            'No examples, computed numbers, answers, result-bearing variable names or implementation instructions. '
            'Only constants present in the original problem are permitted. The function is named solve_unit.'
        )
        error = None
        for attempt in range(2):
            result = self.client.chat.completions.create(model=self.model,
                messages=[{'role': 'user', 'content': prompt}], temperature=0.,
                response_format={'type': 'json_object'}, max_tokens=1024)
            self.teacher_tokens += getattr(getattr(result, 'usage', None), 'total_tokens', 0) or 0
            try:
                record = json.loads(result.choices[0].message.content)
                if set(record) != TEACHER_FIELDS:
                    raise ValueError('unexpected teacher contract fields')
                public = dict(record, unit_id='schema-check', function_name='solve_unit', response_format='Thought and Code')
                validate_contract(public, live_in)
                return record
            except (ValueError, KeyError, TypeError) as exc:
                error = exc
        raise ValueError('contract generation failed schema validation after one retry') from error
