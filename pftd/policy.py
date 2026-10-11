"""Optional PEFT update backend. Heavy dependencies are imported only when used."""
from pathlib import Path
from pftd.grpo import update_group
from pftd.trainer import Rollout


class PeftPolicy:
    def __init__(self, adapter, device='cuda:0', seed=42, sampler=None):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(seed)
        cfg = PeftConfig.from_pretrained(adapter)
        if cfg.r != 64 or cfg.lora_alpha != 128:
            raise ValueError('paper initialization requires rank-64, alpha-128 Agent Distill adapter')
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32)
        self.model = PeftModel.from_pretrained(base, adapter, adapter_name='policy', is_trainable=True)
        self.model.load_adapter(adapter, adapter_name='reference', is_trainable=False)
        self.model.to(device)
        self.device = device
        # Frozen reference is the SAME inherited adapter, not the bare base model.
        self._activate('policy')
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.config.use_cache = False
        self.model.eval()
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=5e-6, betas=(.9, .999), weight_decay=0.)
        self.updates = 0
        self.sampler = sampler

    def _activate(self, name):
        self.model.set_adapter(name)
        for key, p in self.model.named_parameters():
            p.requires_grad_(name == 'policy' and '.policy.' in key)

    def forward(self, segment, reference=False):
        import torch
        self._activate('reference' if reference else 'policy')
        prompt, completion = segment['prompt_ids'], segment['token_ids']
        if not prompt or not completion or len(completion) != len(segment['old_logps']):
            raise ValueError('malformed sampling record')
        tokens = torch.tensor([prompt + completion], device=self.device)
        logits = self.model(input_ids=tokens, use_cache=False).logits[0]
        return logits[len(prompt)-1:len(prompt)+len(completion)-1]

    def sample_messages(self, messages, n, max_new_tokens, stops=('<end_code>',), unit=False):
        import torch
        self._activate('policy')
        self.model.eval()
        prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if unit and len(prompt_ids) > 1536:
            raise ValueError('unit prompt exceeds 1536 tokens; do not silently truncate a contract')
        if len(prompt_ids) + max_new_tokens > 10240:
            raise ValueError('sampling context exceeds maximum length')
        if self.sampler:
            return self.sampler.sample(self, prompt_ids, n, max_new_tokens, stops)
        from transformers import StoppingCriteria, StoppingCriteriaList
        tok = self.tokenizer
        class StopText(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tok.decode(input_ids[0, len(prompt_ids):], skip_special_tokens=False)
                return any(s in text for s in stops)
        result = []
        for _ in range(n):
            with torch.no_grad():
                output = self.model.generate(input_ids=torch.tensor([prompt_ids], device=self.device),
                    attention_mask=torch.ones((1, len(prompt_ids)), dtype=torch.long, device=self.device),
                    max_new_tokens=max_new_tokens, do_sample=True, temperature=1., top_p=1., top_k=0,
                    repetition_penalty=1., return_dict_in_generate=True, output_scores=True, use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([StopText()]))
                ids = output.sequences[0, len(prompt_ids):].tolist()
                old = [float(score[0].float().log_softmax(-1)[t]) for score, t in zip(output.scores, ids)]
            segment = {'prompt_ids': list(prompt_ids), 'token_ids': ids, 'old_logps': old}
            result.append(Rollout(self.tokenizer.decode(ids, skip_special_tokens=True), [segment], len(ids)))
        return result

    def sample(self, prompt, n=4, max_new_tokens=512):
        return self.sample_messages([{'role': 'user', 'content': prompt}], n, max_new_tokens, unit=True)

    def update(self, candidates, rewards):
        if self.sampler:
            self.sampler.sleep()
            self.model.to(self.device)
        result = update_group(self.model, self.optimizer, candidates, rewards,
            self.forward, lambda s: self.forward(s, reference=True), update_index=self.updates)
        self.updates += int(result['updated'])
        return result

    def save(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(destination), selected_adapters=['policy'], safe_serialization=True)
        self.tokenizer.save_pretrained(str(destination))


class VllmSampler:
    """vLLM 0.9.1 sampling with immutable LoRA snapshots and stored behavior logprobs.

    One GPU is shared by moving the update model to CPU while sampling and sleeping
    vLLM before updates. This path needs Linux/CUDA validation on the target machine.
    """
    def __init__(self, base_model, snapshot_dir, seed=42):
        import os
        os.environ.setdefault('VLLM_USE_V1', '0')
        from vllm import LLM
        self.llm = LLM(model=base_model, enable_lora=True, max_lora_rank=64,
            tensor_parallel_size=1, dtype='bfloat16', enforce_eager=True,
            max_model_len=10240, enable_sleep_mode=True, gpu_memory_utilization=.45, seed=seed)
        self.snapshot_dir = Path(snapshot_dir)
        self.version = 0
        self.sleeping = False
        self.sleep()

    def sleep(self):
        if not self.sleeping:
            self.llm.sleep(level=1)
            self.sleeping = True

    def sample(self, policy, prompt_ids, n, max_new_tokens, stops):
        import torch
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        if self.sleeping:
            policy.model.to('cpu')
            torch.cuda.empty_cache()
            self.llm.wake_up()
            self.sleeping = False
        # A new request ID prevents reuse of cached stale adapter weights.
        self.version += 1
        policy.save(self.snapshot_dir)
        request = LoRARequest('policy', self.version, str((self.snapshot_dir / 'policy').resolve()))
        params = SamplingParams(n=n, temperature=1., top_p=1., top_k=-1,
            max_tokens=max_new_tokens, logprobs=1, stop=list(stops), include_stop_str_in_output=True)
        outputs = self.llm.generate([{'prompt_token_ids': prompt_ids}], params, lora_request=request, use_tqdm=False)[0].outputs
        return [Rollout(o.text, [{'prompt_ids': list(prompt_ids), 'token_ids': list(o.token_ids),
            'old_logps': [p[t].logprob for p, t in zip(o.logprobs, o.token_ids)]}], len(o.token_ids)) for o in outputs]
