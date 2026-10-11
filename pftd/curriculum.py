"""Token-based stage quotas, uniform root sampling, mastery and reversible merges."""
from dataclasses import dataclass, field
import random

STAGES = ((0.0, 0.3, 0.6), (0.3, 0.7, 0.3), (0.7, 1.0, 0.0))


@dataclass
class Ledger:
    budget: int = 2_000_000
    unit: int = 0
    full: int = 0
    readiness: int = 0
    stage_unit: list = field(default_factory=lambda: [0, 0, 0])
    stage_full: list = field(default_factory=lambda: [0, 0, 0])

    @property
    def total(self):
        return self.unit + self.full + self.readiness

    @property
    def remaining(self):
        return self.budget - self.total

    @property
    def stage(self):
        b = self.total / self.budget
        return 0 if b < .3 else (1 if b < .7 else 2)

    def charge(self, kind, tokens, stage=None):
        if kind not in {'unit', 'full', 'readiness'} or type(tokens) is not int or tokens < 0:
            raise ValueError('invalid token charge')
        if tokens > self.remaining:
            raise ValueError('generation budget exceeded; cap sampling BEFORE generating')
        stage = self.stage if stage is None else stage
        setattr(self, kind, getattr(self, kind) + tokens)
        if kind != 'readiness':
            getattr(self, 'stage_' + kind)[stage] += tokens

    def want_unit(self):
        stage = self.stage
        share = STAGES[stage][2]
        return share > 0 and self.stage_unit[stage] / max(1, self.stage_unit[stage] + self.stage_full[stage]) < share


@dataclass
class UnitState:
    unit: dict
    weight: float = 1.0
    history: list = field(default_factory=list)
    recent_zero_groups: int = 0

    @property
    def ready(self):
        return len(self.history) >= 2 and min(self.history[-2:]) >= .75


class Curriculum:
    def __init__(self, roots, units, budget=2_000_000, seed=42, variant='full'):
        if variant not in {'full', 'no_merge', 'fixed_schedule', 'no_full', 'outcome_only'}:
            raise ValueError('unknown curriculum variant')
        self.roots = {r['root_id']: r for r in roots if r['split'] == 'train'}
        if not self.roots:
            raise ValueError('no training roots')
        self.units = {u['public']['unit_id']: UnitState(u) for u in units}
        if any(u['private']['split'] != 'train' or u['private']['root_id'] not in self.roots for u in units):
            raise ValueError('unit pool contains a nontraining root')
        self.ledger = Ledger(budget=budget)
        self.rng = random.Random(seed)
        self.variant = variant
        self.next_readiness = .1
        self.merged_pairs = set()
        self.events = []

    def next_prompt(self):
        root = self.rng.choice(list(self.roots.values()))
        active = [s for s in self.units.values() if s.unit['private']['root_id'] == root['root_id'] and s.weight > 0]
        want = self.ledger.want_unit()
        if self.variant == 'outcome_only':
            want = False
        elif self.variant == 'no_full':
            want = True
        if want and active:
            state = self.rng.choices(active, weights=[u.weight for u in active])[0]
            return 'unit', state.unit
        # Uncompilable roots always keep their full-task path.
        return 'full', root

    def readiness_due(self):
        if self.ledger.total / self.ledger.budget < self.next_readiness or self.ledger.stage == 2:
            return False
        while self.next_readiness <= self.ledger.total / self.ledger.budget:
            self.next_readiness += .1
        return True

    def record_readiness(self, uid, rewards):
        # A voided check breaks consecutiveness; never counts as a failed candidate.
        if rewards is None:
            self.units[uid].history.clear()
        elif len(rewards) != 4:
            raise ValueError('readiness requires four fresh candidates')
        else:
            self.units[uid].history.append(sum(r == 1.0 for r in rewards) / 4)

    def record_group(self, uid, rewards):
        state = self.units[uid]
        if rewards is None:
            return
        state.recent_zero_groups = state.recent_zero_groups + 1 if all(r == 0 for r in rewards) else 0
        children = state.unit['private'].get('child_ids', [])
        if children and state.recent_zero_groups >= 3:
            state.weight *= .25
            for child in children:
                self.units[child].weight = 1.0
            state.recent_zero_groups = 0
            self.events.append({'event': 'fallback', 'unit_id': uid, 'tokens': self.ledger.total})

    def maybe_merge(self, compiler):
        if self.variant in {'no_merge', 'outcome_only'} or self.ledger.stage != 1:
            return []
        added = []
        states = list(self.units.items())
        used = set()
        for aid, a in states:
            for bid, b in states:
                pair = (aid, bid)
                pa, pb = a.unit['private'], b.unit['private']
                if aid in used or bid in used or pair in self.merged_pairs or pa['root_id'] != pb['root_id'] or pa['span'][1] != pb['span'][0]:
                    continue
                ready = a.ready and b.ready
                if self.variant == 'fixed_schedule':
                    ready = True
                if not ready:
                    continue
                try:
                    parent = compiler.merge(self.roots[pa['root_id']], a.unit, b.unit)
                except (ValueError, KeyError, TypeError) as exc:
                    self.events.append({'event': 'merge_rejected', 'children': pair, 'reason': str(exc)})
                    continue
                pid = parent['public']['unit_id']
                if pid in self.units:
                    continue
                self.units[pid] = UnitState(parent)
                a.weight *= .25
                b.weight *= .25
                self.merged_pairs.add(pair)
                used.update(pair)
                added.append(parent)
                self.events.append({'event': 'merge', 'unit_id': pid, 'children': pair, 'tokens': self.ledger.total})
        return added
