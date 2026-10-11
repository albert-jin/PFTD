# Agent Distillation source snapshot

This directory retains the upstream implementation used as the basis of PFTD.

Source: GitHub: Nardien/agent-distillation.

Associated paper: Distilling LLM Agent into Small Models with Retrieval and Code Tools, Kang et al. (2025), arXiv: 2505.17612.

The original code includes the smolagents CodeAct agent, trajectory logging and filtering, SFT scripts, evaluation tools, retrieval utilities, and public benchmark inputs. Original source headers and LICENSE are retained. The upstream project acknowledges smolagents, Search-R1 and TRL. This is not an official product of KRAFTON Inc. or DeepAuto.ai.

The PFTD-specific implementation and instructions are in the repository root. The added exps_research/unified_framework/run_pftd.py entry point forwards to that implementation. The original upstream training scripts have their own hardware assumptions; PFTD's configuration uses one A100 80GB.

This README has been replaced by a concise attribution note to keep release documentation anonymous and free of hyperlinks. Refer to docs/UPSTREAM.md in the repository root for the modification inventory.
