"""One shared-policy training loop. Backends never receive private unit records."""
from dataclasses import dataclass
import json
from pathlib import Path
from pftd.contracts import render_prompt
from pftd.grpo import advantages


@dataclass
class Rollout:
    text: str
    segments: list
    generated_tokens: int
    answer: object = None
    failure: str | None = None


class Trainer:
    """Policy interface: sample(prompt,n,max_new_tokens), update(candidates,rewards), save(path).

    full_rollout(question, n, token_cap) -> list[Rollout] must run the ORIGINAL
    CodeAct agent, with observations from the student's executions only. It may
    not receive a teacher trace or reference answer. grade(root,answer) lives on
    the checker side. This dependency injection keeps generation and grading apart.
    """
    def __init__(self, policy, curriculum, compiler, checker, full_rollout, grade, log_path=None):
        self.policy, self.curriculum = policy, curriculum
        self.compiler, self.checker = compiler, checker
        self.full_rollout, self.grade = full_rollout, grade
        self.log_path = Path(log_path) if log_path else None
        self.updates = 0

    def log(self, event):
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')

    def sample_unit(self, unit, readiness=False):
        c = self.curriculum
        stage = c.ledger.stage
        cap = min(512, c.ledger.remaining // 4)
        if cap < 1:
            return None
        rollouts = self.policy.sample(render_prompt(unit['public']), n=4, max_new_tokens=cap)
        if len(rollouts) != 4:
            raise RuntimeError('sampler returned an incomplete group')
        tokens = sum(r.generated_tokens for r in rollouts)
        c.ledger.charge('readiness' if readiness else 'unit', tokens, stage)
        if tokens == 0:
            raise RuntimeError('sampler made no progress')
        rewards, failure = self.checker.group([r.text for r in rollouts], unit['public'], unit['private'], readiness)
        return rollouts, rewards, failure

    def run(self, output, max_groups=None):
        c = self.curriculum
        groups = 0
        while c.ledger.remaining >= 4 and (max_groups is None or groups < max_groups):
            kind, item = c.next_prompt()
            if kind == 'unit':
                sampled = self.sample_unit(item)
                if sampled is None:
                    break
                rollouts, rewards, failure = sampled
                c.record_group(item['public']['unit_id'], rewards)
            else:
                stage = c.ledger.stage
                # Reserve a bounded total cap across all steps of all candidates.
                cap = min(4 * 5 * 1024, c.ledger.remaining)
                rollouts = self.full_rollout(item['question'], 4, cap)
                tokens = sum(r.generated_tokens for r in rollouts)
                c.ledger.charge('full', tokens, stage)
                if not tokens or len(rollouts) != 4:
                    raise RuntimeError('full-task sampler must return a nonempty four-candidate group')
                try:
                    if any(r.failure for r in rollouts):
                        raise RuntimeError(next(r.failure for r in rollouts if r.failure))
                    rewards = [float(self.grade(item, r.answer)) for r in rollouts]
                    failure = None
                except Exception as exc:
                    rewards, failure = None, 'full_task_grader_failure: ' + str(exc)
            status = {'updated': False, 'reason': 'checker_failure' if failure else 'zero_variance_group'}
            if advantages(rewards) is not None:
                status = self.policy.update([r.segments for r in rollouts], rewards)
                self.updates += int(status['updated'])
            groups += 1
            self.log({'event': 'group', 'kind': kind, 'tokens': c.ledger.total,
                'rewards': rewards, 'failure': failure, **status})
            if c.readiness_due():
                for uid, state in list(c.units.items()):
                    if state.weight <= 0 or c.ledger.stage == 2:
                        continue
                    sampled = self.sample_unit(state.unit, readiness=True)
                    if sampled is None:
                        break
                    _, scores, error = sampled
                    c.record_readiness(uid, scores)
                    self.log({'event': 'readiness', 'unit_id': uid, 'tokens': c.ledger.total, 'rewards': scores, 'failure': error})
                parents = c.maybe_merge(self.compiler)
                if parents:
                    from pftd.compiler import save_units
                    save_units(parents, Path(output) / 'compiled')
        # Never select by test/dev accuracy; the last policy is the released checkpoint.
        self.policy.save(Path(output) / 'final')
        state = {'tokens': c.ledger.total, 'unused_budget': c.ledger.remaining,
            'unit_tokens': c.ledger.unit, 'full_tokens': c.ledger.full,
            'readiness_tokens': c.ledger.readiness, 'updates': self.updates, 'groups': groups,
            'events': c.events, 'stopped_by_group_limit': max_groups is not None and groups >= max_groups}
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / 'training_summary.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        return state
