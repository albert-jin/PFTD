# Upstream provenance

- Source: GitHub: Nardien/agent-distillation
- Imported source directory: `vendor/agent_distillation/`
- License: Apache-2.0, preserved in `vendor/agent_distillation/LICENSE`.

The code snapshot includes the original smolagents fork, prompts, tools, evaluation scripts, training scripts and released benchmark inputs. No upstream Git history is rewritten. The PFTD repository's initial commit is retained.

Local additions/changes inside the snapshot:

- `exps_research/unified_framework/run_pftd.py`: integration entry point forwarding to the PFTD training command.
- `README.md`: replaced by a short hyperlink-free upstream attribution and usage note.

All other PFTD behavior is implemented in the top-level `pftd/` package, keeping the upstream source separate from the method additions. The bridge reuses the original short CodeAct prompt and instruction suffix. It suppresses the upstream final-answer fallback model call after the five-step limit, so a hidden sixth call cannot escape the token budget. This override is in `pftd/upstream.py`, not in the original baseline runner.

The upstream scripts may assume multiple GPUs, contain example server addresses, or expect a separately installed retriever. Those are upstream baseline settings; PFTD's own configuration targets one A100 80GB. No credentials, user-specific model directories or private experiment files are included by the PFTD additions.
