"""Thin bridge to the Agent Distillation CodeAct implementation."""
from pathlib import Path
import sys
from pftd.trainer import Rollout

UPSTREAM = Path(__file__).resolve().parents[1] / 'vendor' / 'agent_distillation'


def activate_upstream():
    for path in (UPSTREAM, UPSTREAM / 'src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def full_task_rollout(policy, tools_factory, executor_type='docker', executor_kwargs=None):
    """Return a generator using the original short prompt and tool formatting.

    The caller supplies its original retrieval tools. No private root fields enter
    this closure. Use an isolated machine for inherited execution environments.
    """
    activate_upstream()
    from smolagents import CodeAgent
    from smolagents.models import ChatMessage, get_clean_message_list, remove_tool_call_from_messages, tool_role_conversions
    instruction = "\n\nIMPORTANT: Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail. For math problems that are not multiple-choice, always output the final answer using LaTeX \\boxed{} format. Provide the exact value (e.g., \\boxed{\\frac{9}{14}}), not a decimal approximation (e.g., \\boxed{0.642857})."

    def run(question, n, token_cap):
        results = []
        per_candidate = token_cap // n
        for _ in range(n):
            segments, texts = [], []
            class RecordedModel:
                model_id = 'pftd-policy'
                last_input_token_count = 0
                last_output_token_count = 0
                def __call__(self, messages, stop_sequences=None, **kwargs):
                    remaining = per_candidate - sum(len(s['token_ids']) for s in segments)
                    if remaining < 1:
                        raise RuntimeError('candidate generation cap reached')
                    cleaned = get_clean_message_list(remove_tool_call_from_messages(messages), role_conversions=tool_role_conversions, flatten_messages_as_text=True)
                    out = policy.sample_messages(cleaned, 1, min(1024, remaining), stops=tuple(stop_sequences or []))[0]
                    segments.extend(out.segments)
                    texts.append(out.text)
                    self.last_input_token_count = len(out.segments[0]['prompt_ids'])
                    self.last_output_token_count = out.generated_tokens
                    return ChatMessage(role='assistant', content=out.text)
                def get_token_counts(self):
                    return {'input_token_count': self.last_input_token_count, 'output_token_count': self.last_output_token_count}
            class FiveStepAgent(CodeAgent):
                def _handle_max_steps_reached(self, task, images, step_start_time):
                    # Upstream otherwise makes a sixth, unbudgeted model call.
                    return None
            agent = FiveStepAgent(tools=tools_factory(), model=RecordedModel(), max_steps=5,
                additional_authorized_imports=['numpy', 'sympy', 'numpy.linalg'], verbosity_level=0,
                executor_type=executor_type, executor_kwargs=executor_kwargs or {})
            answer = None
            failure = None
            try:
                answer = agent.run(question + instruction, use_short_system_message=True)
            except Exception as exc:
                # Budget exhaustion is a candidate truncation, not a hidden sixth call.
                # Other environment/model failures void the group after token accounting.
                used = sum(len(s['token_ids']) for s in segments)
                if used < per_candidate:
                    failure = type(exc).__name__ + ': ' + str(exc)
            finally:
                cleanup = getattr(agent.python_executor, 'cleanup', None)
                if cleanup:
                    cleanup()
                # Keep partial generations charged, including malformed actions.
                if not segments:
                    raise RuntimeError('agent failed before producing any tokens')
            results.append(Rollout('\n'.join(texts), segments, sum(len(s['token_ids']) for s in segments), answer, failure))
        return results
    return run


def math_grade(root, answer):
    if answer is None:
        return 0
    activate_upstream()
    from exps_research.unified_framework.math_utils.qwen_math_parser import extract_answer
    from exps_research.unified_framework.math_utils.qwen_math_grader import math_equal
    return int(math_equal(extract_answer(str(answer), data_name='math'), str(root['answer'])))


def factual_grade(model):
    activate_upstream()
    from exps_research.unified_framework.score_answers import evaluate_factual_answer
    def grade(root, answer):
        if answer is None:
            return 0
        # Reuse the original judging prompt; model credentials remain outside the repo.
        return evaluate_factual_answer(model, str(answer), str(root['answer']), root['question'])['score']
    return grade
