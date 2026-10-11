"""Optional full training assembly; pure core modules do not require CUDA."""
import json
import os
from pathlib import Path
from pftd.checker import Checker
from pftd.compiler import Compiler, validate_splits
from pftd.curriculum import Curriculum
from pftd.sandbox import Sandbox
from pftd.trainer import Trainer


def train(args):
    roots = [json.loads(line) for line in Path(args.roots).read_text(encoding='utf-8').splitlines() if line.strip()]
    validate_splits(roots)
    compiled = Path(args.compiled)
    units = []
    for path in sorted((compiled / 'public').glob('*.json')):
        private = compiled / 'private' / path.name
        units.append({'public': json.loads(path.read_text(encoding='utf-8')), 'private': json.loads(private.read_text(encoding='utf-8'))})
    if not units and args.variant != 'outcome_only':
        raise ValueError('no compiled units; use the compiler or explicitly choose outcome_only')
    from pftd.policy import PeftPolicy, VllmSampler
    from pftd.upstream import activate_upstream, full_task_rollout, math_grade, factual_grade
    activate_upstream()
    sampler = None
    if args.sampler == 'vllm':
        from peft import PeftConfig
        base = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
        sampler = VllmSampler(base, Path(args.output) / 'sampling_adapter')
    policy = PeftPolicy(args.adapter, sampler=sampler)
    if args.tools == 'wikipedia':
        from smolagents import WikipediaRetrieverTool
        tools = lambda: [WikipediaRetrieverTool()]
    else:
        tools = lambda: []
    qa_grade = None
    if any(r.get('task_type') == 'qa' for r in roots if r['split'] == 'train'):
        from smolagents import OpenAIServerModel
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('factual QA judge requires OPENAI_API_KEY, never a committed credential')
        qa_grade = factual_grade(OpenAIServerModel(model_id='gpt-4o-mini', api_key=key))
    def grade(root, answer):
        return qa_grade(root, answer) if root.get('task_type') == 'qa' else math_grade(root, answer)
    sandbox = Sandbox()
    curriculum = Curriculum(roots, units, budget=args.budget, variant=args.variant)
    trainer = Trainer(policy, curriculum, Compiler(sandbox), Checker(sandbox, mode=args.reward, answer_grader=grade),
        full_task_rollout(policy, tools), grade, Path(args.output) / 'events.jsonl')
    return trainer.run(args.output)
