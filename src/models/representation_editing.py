"""BitFit, RED and LoReFT baselines on the Whisper decoder.

Each method is installed on a loaded model with `install_intervention(model, peft_args)`
(new modules are attached to the decoder and applied through forward hooks), then
`mark_trainable(model, keys)` freezes everything except the method's parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class BaseIntervention(ABC):
    """Installs interventions on a Whisper model.

    Subclasses set `param_key`, a substring matched by `mark_trainable()` on
    parameter names (e.g. 'red_edit').
    """

    param_key: str = ""

    def __init__(self, config):
        self.config = config
        self._hooks: List = []
        self._owned: List[nn.Module] = []

    @abstractmethod
    def install(self, model: nn.Module) -> None:
        """Wire up hooks and attach any auxiliary modules to `model`."""

    def post_dtype_cast(self, model: nn.Module) -> None:
        """Called after the model is cast to the inference dtype."""
        pass


def mark_trainable(model: nn.Module, param_keys: Iterable[str]) -> int:
    """Freeze everything, then re-enable params whose name contains any key.

    Returns the number of parameter tensors marked trainable.
    """
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    keys = tuple(param_keys)
    for n, p in model.named_parameters():
        if any(k in n for k in keys):
            p.requires_grad = True
            count += 1
    return count


class BitFit(BaseIntervention):
    """BitFit (Zaken et al., 2021): bias-only tuning of decoder cross-attention.

    For each targeted decoder layer, encoder_attn.{q,v,out}_proj.bias stay trainable
    (k_proj has no bias in Whisper). No new parameters or hooks.
    """

    # BitFit selects native tensors by full name through `param_keys`; the
    # class-level `param_key` never matches a Whisper parameter.
    param_key = "__bitfit_unused__"
    _BIAS_SUFFIXES = ("q_proj.bias", "v_proj.bias", "out_proj.bias")

    def __init__(self, config):
        super().__init__(config)
        self.param_keys: List[str] = []

    def install(self, model: nn.Module) -> None:
        if getattr(self.config, "to_encoder", False):
            raise ValueError("BitFit is decoder-only; set to_encoder=False.")
        if not getattr(self.config, "to_decoder", True):
            raise ValueError("BitFit expects to_decoder=True.")

        decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model.decoder
        layers: List[nn.Module] = list(decoder.layers)

        target_idx_1based = self.config.peft_decoder_layers or list(range(1, len(layers) + 1))
        target_idx = sorted({i - 1 for i in target_idx_1based if 1 <= i <= len(layers)})
        if not target_idx:
            raise ValueError(
                f"BitFit: peft_decoder_layers={target_idx_1based} has no entries in range [1..{len(layers)}]."
            )

        base_prefix = getattr(model, "base_model_prefix", "model")
        named_params = dict(model.named_parameters())
        selected: List[str] = []

        for layer_i in target_idx:
            if getattr(layers[layer_i], "encoder_attn", None) is None:
                raise ValueError(f"BitFit: decoder layer {layer_i + 1} has no encoder_attn module.")
            for suffix in self._BIAS_SUFFIXES:
                full_name = f"{base_prefix}.decoder.layers.{layer_i}.encoder_attn.{suffix}"
                if full_name not in named_params:
                    raise ValueError(f"BitFit: expected parameter '{full_name}' was not found.")
                selected.append(full_name)

        self.param_keys = list(selected)

        bias_sites = [s.replace(".bias", "") for s in self._BIAS_SUFFIXES]
        print(
            f"[BitFit] decoder cross-attn bias-only "
            f"layers(1-based)={[i + 1 for i in target_idx]} "
            f"biases={bias_sites} trainable_tensors={len(self.param_keys)}"
        )


class _REDLayerEdit(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Zero-init: identity edit at start.
        self.red_edit_gamma = nn.Parameter(torch.zeros(d_model))
        self.red_edit_beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + self.red_edit_gamma) + self.red_edit_beta


class RED(BaseIntervention):
    """RED (Wu et al., 2024): per-channel affine edit of the decoder FFN output.

    Each targeted decoder layer's `fc2` output becomes h * (1 + gamma) + beta,
    with gamma, beta in R^d. The attention branch and residual stream are unchanged.
    """

    param_key = "red_edit"

    def install(self, model: nn.Module) -> None:
        if not getattr(self.config, "to_decoder", True):
            raise ValueError("RED currently only supports decoder-side editing; set to_decoder=True.")

        decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model.decoder
        layers: List[nn.Module] = list(decoder.layers)
        d_model = model.config.d_model

        target_idx_1based = self.config.peft_decoder_layers or list(range(1, len(layers) + 1))
        target_idx = sorted({i - 1 for i in target_idx_1based if 1 <= i <= len(layers)})
        if not target_idx:
            raise ValueError(
                f"RED: peft_decoder_layers={target_idx_1based} has no entries in range [1..{len(layers)}]."
            )

        edits = nn.ModuleList([_REDLayerEdit(d_model) for _ in target_idx])
        decoder.red_edits = edits
        self._owned.append(edits)

        def make_hook(edit: _REDLayerEdit):
            def hook(_fc2, _inputs, output):
                return edit(output)

            return hook

        for edit, layer_i in zip(edits, target_idx):
            layer = layers[layer_i]
            if not hasattr(layer, "fc2"):
                raise ValueError(f"RED: decoder layer {layer_i + 1} has no 'fc2' FFN projection to hook.")
            h = layer.fc2.register_forward_hook(make_hook(edit))
            self._hooks.append(h)

        print(
            f"[RED] installed on decoder FFN outputs (fc2) "
            f"layers(1-based)={[i + 1 for i in target_idx]} d_model={d_model}"
        )


def _parse_reft_positions(positions: Optional[Sequence[str]]) -> List[Tuple[str, int]]:
    """Parse 'p<k>'/'s<k>' specs into (kind, k) tuples. Empty means every position."""
    if not positions:
        return []
    parsed: List[Tuple[str, int]] = []
    for spec in positions:
        s = str(spec).strip().lower()
        if len(s) < 2 or s[0] not in ("p", "s") or not s[1:].isdigit() or int(s[1:]) < 1:
            raise ValueError(f"Invalid ReFT position spec '{spec}'. Use 'p<k>' or 's<k>' with k>=1.")
        parsed.append((s[0], int(s[1:])))
    return parsed


class _LoReFTIntervention(nn.Module):
    def __init__(self, d_model: int, rank: int, positions: Sequence[str] | None = None):
        super().__init__()
        if rank <= 0 or rank > d_model:
            raise ValueError(f"LoReFT rank must be in [1, d_model={d_model}], got rank={rank}.")
        self.rank = rank
        self.positions: List[Tuple[str, int]] = _parse_reft_positions(positions)
        # R: orthonormal rows (r, d), stored at `reft_R.parametrizations.weight.original`.
        self.reft_R = nn.Linear(d_model, rank, bias=False)
        torch.nn.utils.parametrizations.orthogonal(self.reft_R)
        self.reft_W = nn.Linear(d_model, rank, bias=True)
        # Identity-init: W <- R, b <- 0, so R^T(Wh + b - Rh) = 0 at the first forward.
        with torch.no_grad():
            self.reft_W.weight.copy_(self.reft_R.weight)
            self.reft_W.bias.zero_()
        # Generation step counter for prefix positions; reset per utterance.
        self.register_buffer("gen_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def reset_gen_step(self) -> None:
        self.gen_step.zero_()

    def _position_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.positions:
            return None
        mask = torch.zeros(T, device=device, dtype=dtype)
        for kind, k in self.positions:
            if kind == "p":
                mask[: min(k, T)] = 1.0
            else:  # "s"
                mask[max(0, T - k) :] = 1.0
        return mask

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        T = h.shape[-2]
        # The intervention may stay in fp32 while the model runs in fp16 (see
        # post_dtype_cast): compute in the parameter dtype and cast back.
        param_dtype = self.reft_W.weight.dtype
        orig_dtype = h.dtype
        h_work = h.to(param_dtype) if orig_dtype != param_dtype else h

        R_w = self.reft_R.weight  # (r, d), orthonormal rows
        Rh = h_work @ R_w.transpose(-1, -2)  # (..., r)
        Wh_b = self.reft_W(h_work)  # (..., r)
        delta = (Wh_b - Rh) @ R_w  # (..., d)

        if not self.positions:
            out = h_work + delta
            return out.to(orig_dtype) if orig_dtype != param_dtype else out

        # Generation: one token per step, so only prefix specs can fire.
        if T == 1 and not self.training:
            step = int(self.gen_step.item())
            active = any(kind == "p" and step < k for kind, k in self.positions)
            self.gen_step += 1
            if not active:
                return h
            out = h_work + delta
            return out.to(orig_dtype) if orig_dtype != param_dtype else out

        # Teacher forcing: per-position mask over T.
        mask = self._position_mask(T, h_work.device, h_work.dtype)
        broadcast_shape = [1] * (h_work.dim() - 2) + [T, 1]
        out = h_work + mask.view(*broadcast_shape) * delta
        return out.to(orig_dtype) if orig_dtype != param_dtype else out


class LoReFT(BaseIntervention):
    """LoReFT (Wu et al., 2024): low-rank edit of each targeted decoder layer output.

        h' = h + mask(t) * R^T (W h + b - R h)

    R in R^{r x d} has orthonormal rows, W in R^{r x d}, b in R^r, r = lora_dim.
    `reft_positions` lists 'p<k>' (first k positions) and 's<k>' (last k positions);
    None/empty edits every position. Under generation only prefix specs fire.
    """

    param_key = "reft_"

    def install(self, model: nn.Module) -> None:
        if not getattr(self.config, "to_decoder", True):
            raise ValueError("LoReFT currently only supports decoder-side editing; set to_decoder=True.")

        decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model.decoder
        layers: List[nn.Module] = list(decoder.layers)
        d_model = model.config.d_model
        rank = int(getattr(self.config, "lora_dim", 8))
        positions = getattr(self.config, "reft_positions", None)

        target_idx_1based = self.config.peft_decoder_layers or list(range(1, len(layers) + 1))
        target_idx = sorted({i - 1 for i in target_idx_1based if 1 <= i <= len(layers)})
        if not target_idx:
            raise ValueError(
                f"LoReFT: peft_decoder_layers={target_idx_1based} has no entries in range [1..{len(layers)}]."
            )

        edits = nn.ModuleList([_LoReFTIntervention(d_model, rank, positions) for _ in target_idx])
        decoder.reft_edits = edits
        self._owned.append(edits)

        def make_hook(edit: _LoReFTIntervention):
            def hook(_layer, _inputs, output):
                if isinstance(output, tuple):
                    h = output[0]
                    return (edit(h),) + tuple(output[1:])
                return edit(output)

            return hook

        for edit, layer_i in zip(edits, target_idx):
            h = layers[layer_i].register_forward_hook(make_hook(edit))
            self._hooks.append(h)

        pos_label = positions if positions else "all"
        print(
            f"[LoReFT] installed on decoder layers (1-based): {[i + 1 for i in target_idx]}  "
            f"d_model={d_model}  rank={rank}  positions={pos_label}"
        )

    def post_dtype_cast(self, model: nn.Module) -> None:
        """Keep LoReFT in fp32 when the model is cast to fp16: the orthogonal
        parametrization uses householder_product, which has no fp16 CUDA kernel."""
        for m in self._owned:
            for sub in m.modules():
                if isinstance(sub, _LoReFTIntervention):
                    sub.float()


_REGISTRY = {
    "bitfit": BitFit,
    "red": RED,
    "reft": LoReFT,
    "loreft": LoReFT,
}


def build_intervention(name: str, peft_args) -> BaseIntervention:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown intervention '{name}'. Registered: {sorted(_REGISTRY)}.")
    return _REGISTRY[name](peft_args)


def install_intervention(model: nn.Module, peft_args) -> BaseIntervention:
    intervention = build_intervention(peft_args.peft_type, peft_args)
    intervention.install(model)
    return intervention
