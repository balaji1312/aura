"""AURA gates and training loss.

- Static gates (`log_g1`, `log_g2`) per head: Hard-Concrete sampling during
  training, deterministic stretched sigmoid at eval. The expected-L0 penalty is
  the closed form in `static_gate_l0_expected`.
- Dynamic gate (`SinkGate`): per-head `sigmoid(w_h . feat + b_h)` over three
  features of the cross-attention probs `p` of shape (B*H, T, S):
    * max_prob     = max_s p_s
    * norm_entropy = H(p) / log(S)
    * shift        = |argmax(p_t) - argmax(p_{t-1})| / (S-1)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer


SINK_GATE_FEATURES = ("max_prob", "norm_entropy", "shift")


def static_gate_l0_expected(
    log_alpha: torch.Tensor,
    temperature: float = 0.33,
    stretch_limits: Tuple[float, float] = (-0.1, 1.1),
) -> torch.Tensor:
    """Closed-form expected L0 for a Hard-Concrete gate: P(gate > 0)."""
    low, high = stretch_limits
    shift = temperature * math.log(-low / high)
    return torch.sigmoid(log_alpha - shift).sum()


def sparsity_lambda(global_step: int, max_steps: int, lambda_max: float, ramp_ratio: float) -> float:
    """Linear ramp from 0 to lambda_max over the first ramp_ratio of optimizer steps, then constant."""
    ramp_steps = ramp_ratio * max_steps
    if ramp_steps <= 0:
        return lambda_max
    return lambda_max * min(1.0, global_step / ramp_steps)


class SinkGate(nn.Module):
    """Per-head dynamic gate over attention-sink features.

    Output shape for a given attention tensor of shape (B*H, T, S):
        (B, T, H, 1), broadcast-multiplied with the static gate.

    Parameters per module: 4*H scalars (3 feature weights + 1 bias per head),
    initialised to zeros so the gate starts at 0.5 for all heads.
    """

    FEATURE_DIM = 3  # max_prob, norm_entropy, shift

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # (H, 3) weights per head, (H, 1) bias per head
        self.feat_w = nn.Parameter(torch.zeros(num_heads, self.FEATURE_DIM))
        self.feat_b = nn.Parameter(torch.zeros(num_heads, 1))

        # Mask over the 3 features (order: max_prob, norm_entropy, shift).
        # 1 = feature contributes; 0 = feature zeroed before the projection.
        # Set by `set_sink_gate_disabled_features`; non-persistent.
        _mask = torch.ones(self.FEATURE_DIM)
        self.register_buffer("_feature_mask", _mask.view(1, 1, 1, self.FEATURE_DIM), persistent=False)

        # Argmax of the previous generation step (tgt_len == 1 per step) for the
        # `shift` feature. Reset at the start of every utterance.
        self.register_buffer("prev_argmax", torch.empty(0), persistent=False)

    def reset_prev_argmax(self) -> None:
        self.prev_argmax = torch.empty(0, device=self.feat_w.device)

    @staticmethod
    def _features(
        attn_probs: torch.Tensor,
        num_heads: int,
        prev_argmax: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the 3-feature tensor from softmax attention probs.

        Args:
            attn_probs: (B*H, T, S) post-softmax, pre-dropout.
            num_heads: H
            prev_argmax: (B*H,) last-step argmax for generation; None under
                teacher forcing (shift is built from the tensor itself).
        Returns:
            feats: (B, T, H, 3)
            cur_argmax: (B*H,) argmax of the last time step.
        """
        bsz_times_heads, tgt_len, src_len = attn_probs.shape
        bsz = bsz_times_heads // num_heads

        # 1) max_prob: (B*H, T)
        max_prob, argmax = attn_probs.max(dim=-1)

        # 2) norm_entropy: (B*H, T). Computed in fp32: under fp16 non-peak probs
        # can underflow to 0 and 0 * log(0) gives NaN. Cast back to the probs dtype.
        eps = 1e-9
        probs_fp32 = attn_probs.float()
        entropy = -(probs_fp32 * torch.log(probs_fp32 + eps)).sum(dim=-1)
        norm_entropy = (entropy / math.log(max(src_len, 2))).to(attn_probs.dtype)

        # 3) shift: (B*H, T), dtype-matched to attn_probs.
        probs_dtype = attn_probs.dtype
        denom = max(src_len - 1, 1)
        if tgt_len >= 2:
            # Teacher forcing or multi-token pass: in-tensor argmax deltas.
            diff = (argmax[:, 1:] - argmax[:, :-1]).abs().to(probs_dtype)
            shift_rest = diff / denom
            if prev_argmax is not None and prev_argmax.numel() == bsz_times_heads:
                shift_first = (
                    (argmax[:, 0] - prev_argmax).abs().to(probs_dtype) / denom
                ).unsqueeze(-1)
            else:
                shift_first = torch.zeros(bsz_times_heads, 1, device=attn_probs.device, dtype=probs_dtype)
            shift = torch.cat([shift_first, shift_rest], dim=1)
        else:
            # Single-step generation.
            if prev_argmax is not None and prev_argmax.numel() == bsz_times_heads:
                shift = (
                    (argmax[:, 0] - prev_argmax).abs().to(probs_dtype) / denom
                ).unsqueeze(-1)
            else:
                shift = torch.zeros(bsz_times_heads, 1, device=attn_probs.device, dtype=probs_dtype)

        # Stack: (B*H, T, 3)
        feats = torch.stack([max_prob, norm_entropy, shift], dim=-1)

        # Reshape to (B, T, H, 3) for broadcasting against the static gate.
        feats = feats.view(bsz, num_heads, tgt_len, 3).permute(0, 2, 1, 3).contiguous()

        cur_argmax = argmax[:, -1].detach()

        return feats, cur_argmax

    def forward(
        self,
        attn_probs: torch.Tensor,
        is_generation_step: bool = False,
    ) -> torch.Tensor:
        """Return gate of shape (B, T, H, 1) in [0, 1].

        During generation (tgt_len=1), `is_generation_step=True` feeds the
        previous-step argmax into the `shift` feature.
        """
        prev = self.prev_argmax if (is_generation_step and self.prev_argmax.numel() > 0) else None
        feats, cur_argmax = self._features(attn_probs, self.num_heads, prev)

        if feats.dtype != self.feat_w.dtype:
            feats = feats.to(self.feat_w.dtype)

        feats = feats * self._feature_mask.to(feats.device, feats.dtype)

        # Per-head linear: (B, T, H, 3) . (H, 3) + (H, 1) -> (B, T, H, 1)
        logit = torch.einsum("bthf,hf->bth", feats, self.feat_w).unsqueeze(-1)
        logit = logit + self.feat_b.view(1, 1, self.num_heads, 1)
        gate = torch.sigmoid(logit)

        if is_generation_step:
            self.prev_argmax = cur_argmax

        return gate


def set_sink_gate_disabled_features(model: nn.Module, names: Optional[Iterable[str]]) -> int:
    """Zero the listed features in every SinkGate of `model` (inference-time ablation).

    Returns the number of SinkGate modules updated.
    """
    disabled = set(names or [])
    unknown = disabled - set(SINK_GATE_FEATURES)
    if unknown:
        raise ValueError(
            f"Unknown SinkGate feature(s) {sorted(unknown)}; valid: {', '.join(SINK_GATE_FEATURES)}"
        )
    count = 0
    for module in model.modules():
        if isinstance(module, SinkGate):
            with torch.no_grad():
                module._feature_mask.fill_(1.0)
                for i, feat in enumerate(SINK_GATE_FEATURES):
                    if feat in disabled:
                        module._feature_mask[..., i] = 0.0
            count += 1
    return count


class AuraSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with the expected-L0 penalty on the static gates.

    The SinkGate gets no direct sparsity loss; it is shaped by the task loss only.
    """

    def __init__(
        self,
        *args,
        lambda_max: float = 0.1,
        ramp_ratio: float = 0.1,
        log_every: int = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_max = lambda_max
        self.ramp_ratio = ramp_ratio
        self.gated_lambda: float = 0.0
        self._log_every = log_every

    def _unwrap(self, model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # L = L_task + lambda_t * (1/N) * sum_h [ P(g1_h > 0) + P(g2_h > 0) ],
        # N = number of static gates attached to the model.
        g_l0_sum = torch.tensor(0.0, device=loss.device)
        n_gates = 0
        ghost = torch.tensor(0.0, device=loss.device)

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            ghost = ghost + 0.0 * p.sum()  # keeps every trainable tensor in the graph for DDP
            if ("log_g1" in n) or ("log_g2" in n):
                g_l0_sum = g_l0_sum + static_gate_l0_expected(p)
                n_gates += p.numel()

        max_steps = self.args.max_steps if self.args.max_steps > 0 else self.state.max_steps
        self.gated_lambda = sparsity_lambda(self.state.global_step, max_steps, self.lambda_max, self.ramp_ratio)

        if self.gated_lambda != 0 and n_gates > 0:
            reg = self.gated_lambda * g_l0_sum / n_gates
            loss = loss + reg

        loss = loss + ghost

        if model.training and self.state.global_step % self._log_every == 0:
            self._log_jola_metrics(self._unwrap(model))

        return (loss, outputs) if return_outputs else loss

    def _log_jola_metrics(self, real_model: nn.Module) -> None:
        logs = {"jola/sparsity_lambda": float(self.gated_lambda)}

        gate_means = []
        feat_w_norms = []
        for mod in real_model.modules():
            sg = getattr(mod, "sink_gate", None)
            if sg is not None and hasattr(sg, "feat_w"):
                feat_w_norms.append(sg.feat_w.detach().norm(dim=-1).mean().item())
                last = getattr(mod, "last_sink_gate_mean", None)
                if last is not None:
                    gate_means.append(float(last))

        if gate_means:
            logs["jola/sink_gate_mean"] = sum(gate_means) / len(gate_means)
        if feat_w_norms:
            logs["jola/sink_feat_w_norm"] = sum(feat_w_norms) / len(feat_w_norms)

        self.log(logs)
