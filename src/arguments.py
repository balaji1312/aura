from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

@dataclass
class WhisperModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention implementation, 'eager' or 'sdpa'. LoRA and JoLA/AURA need 'eager'."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_path: dict = field(
        default=None, metadata={"help": "Paths to the training datasets"}
    )
    dev_data_path: dict = field(
        default=None, metadata={"help": "Paths to the development datasets. Not needed when evaluation_strategy is 'no'."}
    )
    streaming: bool = field(
        default=False, metadata={"help": "Map-style or iterable datasets."}
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="sentence",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'sentence'"},
    )
    strip_label_text: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Strip whitespace from the normalized transcript before tokenization. "
                "Default: stripped for PEFT methods, not for full fine-tuning."
            )
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language of the label prefix for multilingual models, e.g. `en`. English-only (.en) models "
                "keep their own prefix."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )

@dataclass
class PEFTArguments:
    """
    Arguments pertaining to parameter efficient training: lora, jola, aura, bitfit, red and loreft.
    """

    peft_type: str = field(
        default=None, metadata={"help": "peft type: lora, jola, aura, bitfit, red or loreft. None for full fine-tuning."}
    )
    lora_dim: int = field(
        default=8, metadata={"help": "The rank for LoRA and LoReFT."}
    )
    lora_alpha: int = field(
        default=128, metadata={"help": "Alpha for scaling factor."}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout rate in Lora layers."}
    )
    peft_encoder_layers: List[int] = field(
        default=None, metadata={"help": "layers to use peft in encoder, 1->n are encoder layers. All methods here edit the decoder only."}
    )
    peft_decoder_layers: List[int] = field(
        default=None, metadata={"help": "layers to use peft in decoder,  1->n are decoder layers"}
    )
    to_encoder: bool = field(
        default=False, metadata={"help": "use peft to encoder. Not supported; AURA+Enc trains the encoder with jola_encoder_train."}
    )
    to_decoder: bool = field(
        default=True, metadata={"help": "use peft to decoder."},
    )
    jola_target_modules: List[str] = field(
        default_factory=lambda: ["encoder_attn"],
        metadata={"help": "Decoder attention modules edited by JoLA/AURA: 'self_attn', 'encoder_attn'. Default is cross-attention."}
    )
    jola_gate_hp: Dict[str, Any] = field(
        default_factory=lambda: {'temperature': 0.33, 'stretch_limits': [-0.1, 1.1], 'eps': 1e-6}, metadata={"help": "hyperparameters for jola gates, temperature, stretch_limits, eps"},
    )
    jola_gate_type: str = field(
        default="static", metadata={"help": "type of jola gate: 'static' (Hard-Concrete, JoLA) or 'dynamic' (Hard-Concrete x SinkGate, AURA)."},
    )
    jola_sparsity_weight: float = field(
        default=0.1, metadata={"help": "final weight of the expected-L0 sparsity loss on the Hard-Concrete gates"},
    )
    jola_sparsity_ramp_ratio: float = field(
        default=0.1, metadata={"help": "fraction of optimizer steps over which the sparsity weight ramps linearly from 0"},
    )
    jola_encoder_train: bool = field(
        default=False, metadata={"help": "also train all encoder parameters (AURA+Enc)"},
    )
    reft_positions: List[str] = field(
        default=None,
        metadata={"help": "ReFT intervention positions: list of 'p<k>' (first k prefix tokens) and/or 's<k>' (last k suffix tokens), e.g. ['p2','s2']. Default None = intervene at every position. Suffix specs only apply under teacher forcing; at generation only prefix specs fire."},
    )

    def __post_init__(self):
        # `aura` is JoLA with the dynamic SinkGate.
        if self.peft_type == "aura":
            self.peft_type = "jola"
            if self.jola_gate_type != "dynamic":
                self.jola_gate_type = "dynamic"

    @classmethod
    def from_saved(cls, d):
        """Rebuild the arguments saved as `peft_config` in a checkpoint's config.json.

        Keys of options that are not part of this code are dropped. Loading fails only if a
        dropped key changed the saved model, since its weights could not be rebuilt here.
        Returns None when there is no PEFT method.
        """
        if isinstance(d, cls):
            d = asdict(d)
        if not d:
            return None
        d = dict(d)
        names = {f.name for f in fields(cls)}
        dropped = {k: d.pop(k) for k in list(d) if k not in names}

        peft_type = d.get("peft_type")
        if peft_type not in (None, "lora", "jola", "aura", "bitfit", "red", "loreft", "reft"):
            raise ValueError(f"peft_type '{peft_type}' is not supported.")
        lora_targets = dropped.get("lora_target_modules")
        if peft_type == "lora" and lora_targets is not None and set(lora_targets) != {"self_attn", "encoder_attn"}:
            raise ValueError(f"LoRA was saved with lora_target_modules={lora_targets}; only both attentions are supported.")
        per_target = dropped.get("jola_gate_type_per_target")
        if peft_type in ("jola", "aura") and per_target:
            gate_type = "dynamic" if peft_type == "aura" else d.get("jola_gate_type", "static")
            targets = d.get("jola_target_modules") or ["encoder_attn"]
            if any(per_target.get(t, gate_type) != gate_type for t in targets):
                raise ValueError(f"JoLA was saved with jola_gate_type_per_target={per_target}, which is not supported.")

        if dropped:
            print(f"peft_config: ignoring saved keys {sorted(dropped)}")
        args = cls(**d)
        return None if args.peft_type is None else args
