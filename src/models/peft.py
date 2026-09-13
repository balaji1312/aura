# Reference code: Loralib from Microsoft
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

import dataclasses
import math

import torch.nn as nn

from models.representation_editing import install_intervention, mark_trainable


class LoRALinear(nn.Module):
    def __init__(self, model, rank, alpha, dropout):
        super(LoRALinear, self).__init__()

        self.rank = rank
        self.lora_alpha = alpha

        if not isinstance(model, nn.Linear):
            raise NotImplementedError("For Linear layer!")

        out_dim, in_dim = model.weight.size()
        self.lora_A = nn.Parameter(model.weight.new_zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(model.weight.new_zeros((out_dim, rank)))
        self.scaling = alpha / rank

        if dropout > 0:
            self.lora_dropout = nn.Dropout(p=dropout)
        else:
            self.lora_dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False


# JoLA/AURA tensors: per-head scale A and bias v, static gates, SinkGate projection.
_JOLA_PARAM_KEYS = (
    "log_g1",
    "log_g2",
    "attn_A",
    "attn_v",
    "sink_gate.feat_w",
    "sink_gate.feat_b",
)


def _is_jola_param(name: str) -> bool:
    return any(k in name for k in _JOLA_PARAM_KEYS)


def mark_only_jola_as_trainable(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    for n, p in model.named_parameters():
        if _is_jola_param(n):
            p.requires_grad = True
            count += 1
    print(f"JoLA Setup: {count} parameter tensors set to trainable.")


def mark_jola_and_encoder_as_trainable(model: nn.Module) -> None:
    """AURA+Enc: AURA parameters plus every encoder parameter (incl. embed_positions)."""
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    for n, p in model.named_parameters():
        if _is_jola_param(n) or n.startswith("model.encoder"):
            p.requires_grad = True
            count += 1
    print(f"JoLA+Encoder Setup: {count} parameter tensors set to trainable.")


def apply_peft(model, peft_args) -> str:
    """Install the method selected by `peft_args` on a loaded model and set the trainable set.

    Returns the method name: full_ft, lora, jola, aura, aura_enc, bitfit, red or loreft.
    """
    peft_type = peft_args.peft_type

    # Full fine-tuning: every parameter stays trainable except the frozen
    # sinusoidal encoder positions.
    if peft_type is None:
        return "full_ft"

    if peft_type in ("lora", "jola"):
        if peft_type == "jola" and peft_args.jola_encoder_train and peft_args.jola_gate_type != "dynamic":
            raise ValueError("jola_encoder_train is only supported for AURA (jola_gate_type='dynamic').")

        model.setup_peft(peft_args)
        if peft_type == "lora":
            mark_only_lora_as_trainable(model)
            method = "lora"
        elif peft_args.jola_encoder_train:
            mark_jola_and_encoder_as_trainable(model)
            method = "aura_enc"
        else:
            mark_only_jola_as_trainable(model)
            method = "aura" if peft_args.jola_gate_type == "dynamic" else "jola"
        return method

    intervention = install_intervention(model, peft_args)

    # Hook methods expose one `param_key` substring; BitFit lists full parameter names.
    trainable_keys = getattr(intervention, "param_keys", None) or [intervention.param_key]
    n_trainable = mark_trainable(model, trainable_keys)
    if n_trainable == 0:
        raise RuntimeError(f"No parameters were marked trainable for peft_type='{peft_type}'.")

    # Stored in config.json so the decoder can re-install the intervention.
    model.config.peft_config = dataclasses.asdict(peft_args)

    return "loreft" if peft_type == "reft" else peft_type
