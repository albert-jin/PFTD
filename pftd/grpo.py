"""Clipped GRPO with exact vocabulary KL, candidate normalization and token masks."""
import math


def advantages(rewards, epsilon=1e-8):
    if rewards is None:
        return None
    if not rewards or any(not math.isfinite(r) for r in rewards):
        raise ValueError('invalid reward group')
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean)**2 for r in rewards) / len(rewards))
    if std == 0:
        return None
    return [(r - mean) / (std + epsilon) for r in rewards]


def exact_kl(logits, reference_logits, chunk_size=4096):
    """KL(pi_theta || pi_theta0), not the sampled-action approximation."""
    import torch
    logits = logits.float()
    reference_logits = reference_logits.detach().float()
    logz = torch.logsumexp(logits, dim=-1, keepdim=True)
    refz = torch.logsumexp(reference_logits, dim=-1, keepdim=True)
    kl = torch.zeros_like(logz[..., 0])
    for start in range(0, logits.shape[-1], chunk_size):
        lp = logits[..., start:start+chunk_size] - logz
        lq = reference_logits[..., start:start+chunk_size] - refz
        kl = kl + (lp.exp() * (lp-lq)).sum(dim=-1)
    return kl


def token_objective(logits, reference_logits, labels, old_logps, mask, advantage, clip=.2, beta=.01, chunk_size=4096):
    """Inputs have aligned token positions, with prompt/observation positions masked.

    Caller sums segment numerators and divides by TOTAL generated tokens of the
    candidate, then averages candidates. This matters for multi-turn agents.
    """
    import torch
    mask = mask.bool()
    if not bool(mask.any()):
        raise ValueError('candidate contains no student-generated tokens')
    z, ref, labels, old = logits[mask], reference_logits[mask], labels[mask], old_logps[mask].detach()
    logps = z.float().log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    ratio = (logps - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    terms = surrogate - beta * exact_kl(z, ref, chunk_size)
    return -terms.sum(), int(mask.sum())


def update_group(model, optimizer, candidates, rewards, forward, reference_forward,
                 learning_rate=5e-6, update_index=0, warmup=20, grad_clip=1.0):
    """Each candidate is a list of exact sampling-context segments.

    Segment fields: prompt_ids, token_ids, old_logps. Only token_ids get loss;
    tool observations occur solely inside prompt_ids. No cross-example packing.
    forward(segment) returns logits at each generated token's preceding position.
    """
    import torch
    adv = advantages(rewards)
    if adv is None:
        return {'updated': False, 'reason': 'checker_failure' if rewards is None else 'zero_variance_group'}
    if len(candidates) != len(adv):
        raise ValueError('candidate/reward group size mismatch')
    model.eval()  # dropout off even though autograd remains enabled
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate * min(1., (update_index + 1) / max(warmup, 1))
    for group in optimizer.param_groups:
        group['lr'] = lr
    loss_total = 0.
    for candidate, a in zip(candidates, adv):
        n = sum(len(segment['token_ids']) for segment in candidate)
        if n == 0:
            raise ValueError('empty sampled candidate')
        for segment in candidate:
            with torch.no_grad():
                ref = reference_forward(segment)
            logits = forward(segment)
            labels = torch.tensor(segment['token_ids'], device=logits.device, dtype=torch.long)
            old = torch.tensor(segment['old_logps'], device=logits.device, dtype=torch.float32)
            numerator, _ = token_objective(logits, ref, labels, old, torch.ones_like(labels, dtype=torch.bool), a)
            loss = numerator / (n * len(candidates))
            loss.backward()
            loss_total += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {'updated': True, 'loss': loss_total, 'gradient_norm': float(norm), 'learning_rate': lr}
