# Implementation notes

## Paper-to-code mapping

| Manuscript mechanism | Implementation |
| --- | --- |
| RBW / Def / Live interfaces | `interfaces.py`: `block`, `interface` |
| Complete-statement spans and at most twelve candidates | `interfaces.py`: `candidate_spans` |
| At most three base units per root | `compiler.py`: `compile_roots` |
| Replay and valid problem contexts | `compiler.py`: `compile_unit` |
| Independent semantic checking and mutation rejection | `compiler.py`: `compile_unit`, `mutate` |
| Public contract whitelist | `contracts.py`: `validate_contract`, `render_prompt` |
| No private values in teacher contract requests | `teacher.py`: `TeacherContractGenerator` |
| Fresh candidate and downstream states | `sandbox.py`, `worker.py`, `checker.py` |
| Fraction of joint local/downstream successes | `checker.py`: `context`, `check` |
| Checker failure voids the group | `checker.py`: `group` |
| Exact chunked vocabulary KL | `grpo.py`: `exact_kl`, `token_objective` |
| Generated-token loss, no observation loss | `grpo.py`: `update_group`; `policy.py` sampling segments |
| Frozen initial Agent Distill policy | `policy.py`: separate policy/reference adapters |
| Root-first sampling, 60/30/0 token shares | `curriculum.py`: `Ledger`, `next_prompt` |
| Four held-aside contexts and two consecutive readiness checks | `curriculum.py`: `record_readiness` |
| Rebuild parent interfaces from the original program | `compiler.py`: `merge` |
| Child down-weighting and all-zero fallback | `curriculum.py`: `maybe_merge`, `record_group` |
| One shared student; final checkpoint | `trainer.py`: `run` |

## Root format

Input JSONL has one object per root:

- `root_id`, `family_id`, `split`: fixed before deriving units. A root or near-duplicate family cannot cross train/dev/test boundaries.
- `question`, `question_template`, `answer`, `task_type`: original question, explicit parameterized question, gold answer and optional `qa` task tag.
- `teacher_correct`, `program`, `turn_ends`: correctness decision, replayable concatenation of action code, and top-level statement boundaries of teacher turns.
- `imports`: declared deterministic libraries.
- `parameter_map`: map root parameter names to top-level literal-assignment statement indices. Only these statements may be parameterized; intermediate values are never edited independently.
- `original_parameters`, `reward_parameters`, `readiness_parameters`: original root values, six reward assignments and four disjoint scheduling assignments.
- `answer_oracle_code`: a separately justified `solve_unit(parameters)` that rejects invalid problem instances and computes their final answers.
- `span_plans`: keys such as `1:3` identify zero-based half-open spans. Each plan contains a public `contract`, independent `oracle_code`, and `oracle_provenance`. Plans for prospective parent spans are needed for recompiled merges.

The compiler can consume a teacher contract generator instead of cached contracts. The generator receives only the question, span code and interface types/names, not later code, reference values or gold answers. Structural JSON failures get one retry. Admission remains fail-closed for semantic, leakage and mutation failures; a rejected contract can be regenerated offline before recompilation.

Independent verification is a required scientific input. The code verifies oracle agreement on every context, but cannot establish the human claim that two algorithms were developed independently. Oracle provenance therefore remains explicit. Missing plans never silently become `semantic_checked` units.

Retrieval trajectories must first freeze retrieved passages as explicit literal inputs and retain the source trace provenance. Live retrieval/network operations are not compiled. This release does not guess retrieval freezes or problem parameterizations from arbitrary logs.

## Public/private boundary

Generation consumes `render_prompt(unit['public'])`. The private record holds the root/family identity, source program, span, reference function, parameterization map, inputs, expected outputs, expected final answers, scheduling contexts, mutation audit and merge ancestry. It is never passed to `policy.sample`.

Candidate workers receive only a function and arguments. Downstream workers replay the prefix afresh, inject the returned value in the ordered live-out interface and execute the suffix. They never copy the teacher's completed globals. Output transfer uses a bounded typed JSON format, not pickle.

Integer, rational, boolean and string semantics are exact. Float tolerances default to relative 1e-9 and absolute 1e-12. Container type, tuple arity, order and duplicates follow the contract. A list with the right sum but the wrong elements is rejected locally; output shape/order errors can also break downstream execution.

## GRPO and sampling records

Each sampled action records its exact `prompt_ids`, `token_ids` and `old_logps`. In a multi-turn agent, each action is a separate segment because the next prompt includes the student's own tool observations. The optimizer recomputes probabilities in exactly these contexts; it does not re-tokenize a reconstructed conversation.

The objective uses population-standard-deviation normalized group rewards with epsilon 1e-8, clipped probability ratios with epsilon 0.2, and KL coefficient 0.01. KL is `sum_v p(v) * (log p(v) - log q(v))` over the complete vocabulary, accumulated in chunks. The loss sums all generated-token contributions of a candidate, divides by that candidate's generated length, then averages candidates. Equal-reward groups and voided groups produce no optimizer step.

The PEFT backend has two copies of the initial adapter: trainable policy and frozen reference. It does not use the bare pretrained model as the reference. LoRA dropout is disabled. AdamW uses betas (0.9, 0.999), no weight decay, a twenty-update warm-up, and gradient norm clipping at 1.0.

## Curriculum details

Unit, full-task and readiness generation have separate ledgers. All three count against the total budget, including invalid candidates and groups that do not update. Readiness generation does not count toward the unit/full rollout-share denominator. Stage boundaries use total consumed budget; rollout deficits are tracked separately within each stage to implement the stated 60/30/0 stage quotas. Groups are indivisible; a group crossing a stage boundary is attributed to its starting stage.

The manuscript's short scheduler listing uses cumulative rollout fractions, while its text states per-stage shares. This implementation follows the per-stage shares. At least four tokens must remain before starting a four-candidate group; a tail of fewer than four tokens is left unused and recorded.

Each readiness estimate measures the fraction of four fresh candidates passing every held-aside context. Two consecutive estimates of at least 0.75 are required. A voided check resets consecutive readiness history without supplying a negative training reward. Merges occur in the middle stage and re-run the compiler on the full parent span. Children retain one-quarter weight. Three all-zero parent groups down-weight the parent and restore its children's weights.

## Execution scope

The AST analysis is intentionally conservative and is confirmed by fresh input-only executions on every supplied context. Loops, branches, try blocks and function definitions remain atomic statements. Unsupported dynamic names, decorators, classes, asynchronous code, unserializable function-valued boundaries, hidden state or ambiguous dependencies reject spans. Library-backed computations are accepted only when boundary values belong to the supported Python value domain.

Docker isolates generated unit code; the Python AST checks alone are not a security boundary. Compilation and checking require Docker. Docker/CUDA integration requires target-environment validation; arbitrary adversarial sandbox escape resistance has not been certified.

The original full-task agent executor and external retriever are integration dependencies, not the private unit checker. Run them in an isolated training environment. Original math-grader package compatibility and factual-judge credentials must be configured separately; a missing or failed grader must never be replaced by a fabricated score.

## Release boundary

Included: core method code, upstream source snapshot, optional training backends, paper settings, and manuscript figures/results for presentation.

Not included: original private traces, original per-root oracle collection, trained PFTD checkpoints, rerun benchmark predictions, the authors' manually audited checker alternatives, or a claim that all manuscript measurements were reproduced by this code release.
