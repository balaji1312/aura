#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys

import datasets
import evaluate
import torch
import yaml
from datasets import DatasetDict, IterableDatasetDict
from datasets.distributed import split_dataset_by_node
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint, is_main_process

sys.path.append(os.environ["rootdir"] + "/src")

from arguments import WhisperModelArguments, DataTrainingArguments, PEFTArguments
from data.whisper_loader import WhisperDataset
from data.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from models.aura import AuraSeq2SeqTrainer
from models.modeling_whisper import WhisperForConditionalGeneration
from models.feature_extractor import WhisperFeatureExtractor
from models.peft import apply_peft

logger = logging.getLogger(__name__)


class SaveProcessorCallback(TrainerCallback):
    """Writes the feature extractor and tokenizer (incl. normalizer.json) into every checkpoint."""

    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.feature_extractor.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)


def main():
    parser = HfArgumentParser((WhisperModelArguments, DataTrainingArguments, PEFTArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".json")):
        # values from the config file, overridden by any --field value given after it
        with open(os.path.abspath(sys.argv[1]), encoding="utf-8") as f:
            args = json.load(f) if sys.argv[1].endswith(".json") else yaml.safe_load(f)
        for action in parser._actions:
            action.default, action.required = argparse.SUPPRESS, False
        args.update(vars(parser.parse_args(sys.argv[2:])))
        model_args, data_args, peft_args, training_args = parser.parse_dict(args)
    else:
        model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    if transformers.__version__ != "4.48.3":
        logger.warning(f"This code was tested with transformers==4.48.3, found {transformers.__version__}.")

    # LoRA and the JoLA/AURA edit exist only in the eager attention
    if peft_args.peft_type in ("jola", "lora") and model_args.attn_implementation != "eager":
        raise ValueError(f"peft_type '{peft_args.peft_type}' requires attn_implementation 'eager'.")

    do_eval = training_args.eval_strategy != "no"
    if do_eval and data_args.dev_data_path is None:
        raise ValueError("dev_data_path is required when evaluation_strategy is not 'no'.")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.warning("checkpoint found, resuming")

    set_seed(training_args.seed)

    # Dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    raw_datasets["train"] = WhisperDataset(data_args.train_data_path, data_args).data
    if do_eval:
        raw_datasets["valid"] = WhisperDataset(data_args.dev_data_path).data

    if data_args.streaming:
        raw_datasets["train"] = raw_datasets["train"].to_iterable_dataset(num_shards=8)
        if do_eval:
            raw_datasets["valid"] = raw_datasets["valid"].to_iterable_dataset(num_shards=8)
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size > 1:
            rank = torch.distributed.get_rank()
            raw_datasets["train"] = split_dataset_by_node(raw_datasets["train"], rank=rank, world_size=world_size)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=training_args.seed, buffer_size=100000)
            if do_eval:
                raw_datasets["valid"] = split_dataset_by_node(raw_datasets["valid"], rank=rank, world_size=world_size)

    assert data_args.audio_column_name in next(iter(raw_datasets.values())).column_names, "missing audio or incorrect audio column name"
    assert data_args.text_column_name in next(iter(raw_datasets.values())).column_names, "missing text or incorrect text column name"

    # Model + tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config.update({"suppress_tokens": model_args.suppress_tokens, "use_cache": False})
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print("PEFT arguments:")
    print(peft_args)

    # Installs the method, sets the trainable parameters and stores peft_config in model.config
    method = apply_peft(model, peft_args)
    print(f"Training method: {method}")
    if peft_args.peft_type is not None:
        training_args.ddp_find_unused_parameters = False

    if training_args.gradient_checkpointing:
        # Trainer enables checkpointing again from training_args at train()
        if training_args.gradient_checkpointing_kwargs is None:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
        model.enable_input_require_grads()
        model.config.use_cache = False

    # Training labels start with the decoder prompt used at decoding:
    # <|sot|><|lang|><|task|><|notimestamps|> for multilingual models and
    # <|sot|><|notimestamps|> for English-only models.
    if getattr(model.generation_config, "is_multilingual", False):
        if data_args.language is None:
            raise ValueError("language must be set for multilingual models.")
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)
        model.config.decoder_input_ids = tokenizer.get_decoder_prompt_ids(language=data_args.language, task=data_args.task)
        model.generation_config.decoder_input_ids = model.config.decoder_input_ids
        model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids
        model.generation_config.suppress_tokens = model.config.suppress_tokens
        model.config.label_prefix = "full"
        model.config.label_language = data_args.language
        model.config.label_task = data_args.task
    else:
        model.config.label_prefix = "short"

    # Resample
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    strip_label_text = data_args.strip_label_text
    if strip_label_text is None:
        strip_label_text = peft_args.peft_type is not None

    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    def prepare_dataset(batch):
        sample = batch[audio_column_name]
        try:
            audio = sample["array"]
            sr = sample["sampling_rate"]
            orig_len = len(audio)
        except Exception:
            audio = np.zeros(1000, dtype=np.float32)
            sr = 16000
            orig_len = 0

        if orig_len < 1000:
            if orig_len == 0:
                audio = np.zeros(1000, dtype=np.float32)
            else:
                pad_width = 1000 - orig_len
                audio = np.pad(audio, (0, pad_width), mode="constant")

        inputs = feature_extractor(audio, sampling_rate=sr, return_attention_mask=forward_attention_mask)
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = orig_len
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # an empty reference (non-speech) becomes prefix + <|endoftext|>
        input_str = tokenizer._normalize(batch[text_column_name])
        if strip_label_text:
            input_str = input_str.strip()
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
        )

    def is_audio_in_length_range(length):
        return length > min_input_length and length <= max_input_length

    vectorized_datasets = vectorized_datasets.filter(is_audio_in_length_range, input_columns=["input_length"])

    metric = evaluate.load("wer") if do_eval and training_args.predict_with_generate else None

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = [s.strip() for s in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
        label_str = [s.strip() for s in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

        nonempty_pred_rate = sum(1 for s in pred_str if len(s) > 0) / max(1, len(pred_str))
        keep = [i for i, r in enumerate(label_str) if len(r) > 0 and r != "<UNK>"]
        if len(keep) == 0:
            return {"wer": 0.0, "nonempty_pred_rate": nonempty_pred_rate}
        wer = metric.compute(
            predictions=[pred_str[i] for i in keep],
            references=[label_str[i] for i in keep],
        )
        return {"wer": wer, "nonempty_pred_rate": nonempty_pred_rate}

    with training_args.main_process_first():
        if is_main_process(training_args.local_rank):
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)
    processor.feature_extractor = feature_extractor

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["valid"] if do_eval else None,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if metric is not None else None,
        callbacks=[SaveProcessorCallback(feature_extractor, tokenizer)],
    )
    if peft_args.peft_type in ("jola", "lora"):
        # L = L_task + lambda * expected L0 of the static gates; lambda ramps linearly from 0
        # to jola_sparsity_weight over the first jola_sparsity_ramp_ratio of the optimizer steps
        trainer = AuraSeq2SeqTrainer(
            lambda_max=peft_args.jola_sparsity_weight,
            ramp_ratio=peft_args.jola_sparsity_ramp_ratio,
            **trainer_kwargs,
        )
    else:
        trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("\n" + "=" * 50)
    print("VERIFYING TRAINABLE PARAMETERS")
    print("=" * 50)
    trainable_params = 0
    all_param = 0
    trainable_names = []
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_names.append(name)
            trainable_params += param.numel()
            print(f" -> TRAINABLE: {name} | Shape: {tuple(param.shape)}")
    print("-" * 50)
    print(f"Trainable Modules Found: {len(trainable_names)}")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"Total Parameters:        {all_param:,}")
    print(f"Trainable %:             {100 * trainable_params / all_param:.4f}%")
    print("=" * 50 + "\n", flush=True)
    assert trainable_params > 0, "ERROR: No parameters set to train."

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
