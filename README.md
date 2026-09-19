# AURA: Uncertainty-Routed Activation Editing for Acoustic Grounding in Speech Foundation Models

## Overview

This repository contains the code and models for our paper **"AURA: Uncertainty-Routed Activation Editing for Acoustic Grounding in Speech Foundation Models"**, accepted at IEEE SLT 2026.

AURA (Activation-editing with Uncertainty-Routed Adaptation) is an ultra-efficient representation-editing method for attention encoder-decoder speech foundation models. The pretrained Whisper model stays frozen. AURA learns sparse scale-and-shift edits for individual decoder cross-attention heads. Static Hard-Concrete gates choose which heads get edited. A token-level dynamic gate then decides how strongly each edit applies, based on three cross-attention uncertainty features: over-concentration (maximum attention probability), diffuse attention (normalized entropy), and abrupt shifts in the attended frame.

We evaluate AURA on non-speech hallucination and on three speech grounding stressors: child speech with imperfect labels (MyST), adult speech with imperfect labels (TED-LIUM 3), and disfluent speech (FluencyBank). On non-speech audio, AURA sharply reduces the hallucination rate of Whisper-large-v3 without identifying hallucination heads beforehand. On the speech corpora, it approaches LoRA WER with a small fraction of LoRA's trainable parameters.

## Folder Structure

Source code for training and decoding models can be found under `/src`:

- `models/aura.py` — AURA gates: static Hard-Concrete head gates, the uncertainty-routed dynamic gate (`SinkGate`), and the trainer with the ramped sparsity penalty
- `models/modeling_whisper.py` — Whisper implementation modified to inject LoRA adapters and JoLA/AURA head edits into the decoder attention layers
- `models/peft.py` — LoRA layers and the selection of trainable parameters for each method
- `models/representation_editing.py` — BitFit, RED, and LoReFT baselines
- `bin/train_asr.py` — training entry point
- `bin/decode_asr.py` — decoding and scoring entry point (WER, hallucination rates, disfluency F1)

`/egs` contains the recipes:

- `MyST`, `tedlium3`, `fluencybank` — speech grounding experiments (WER). Each recipe contains a staged `run_whisper.sh`, the `path.sh`/`cmd.sh` environment files, and training configs `conf/<method>/whisper_<size>_train.yaml` for AURA and the baselines (`lora`, `bitfit`, `red`, `loreft`, `jola`, `aura`) at every model size (`tiny`, `base`, `small`, `medium`, `largev3`).
- `halluc` — non-speech hallucination experiments with Whisper-large-v3. It has the same layout, AURA and JoLA configs, and data preparation scripts under `local/`.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install torch "transformers==4.48.3" datasets evaluate jiwer librosa soundfile pyyaml
   ```

   `wav.scp` entries of the form `flac -c -d -s <file> |` (as in the LibriSpeech preparation) need the `flac` command-line tool.

2. **Prepare data.** Each recipe expects Kaldi-style `wav.scp` and `text` files (one `utt_id` per line) under `egs/<dataset>/data/{train,dev,test}/`. For the non-speech experiments, stage 0 of `egs/halluc/run_whisper.sh` prepares all data:
   - the training set: AudioSet with speech clips removed, DEMAND, and MUSAN music and noise, all with empty transcripts
   - the UrbanSound8K, LibriSpeech test-clean, and test-other evaluation sets

3. **Configure training.** Edit the YAML config for the desired method and model size, e.g. `egs/MyST/conf/aura/whisper_medium_train.yaml`. `peft_type` selects the method: `aura`, `jola`, `lora`, `bitfit`, `red`, or `loreft`. Any field can also be overridden on the command line after the config file, e.g. `train_asr.py conf/aura/whisper_medium_train.yaml --output_dir exp/my_run`.

4. **Run.** From a recipe directory, edit the stage range and the `methods`/`model_sizes` lists in `run_whisper.sh` and run it. The script is staged as zero-shot decoding → fine-tuning → decoding the fine-tuned models. The speech recipes decode every checkpoint on the dev set, then the best dev checkpoint on the test set; `halluc` decodes every saved checkpoint on the evaluation sets.

   ```bash
   cd egs/MyST
   ./run_whisper.sh
   ```

   `decode_asr.py` writes `ref.txt` and `hyp.txt`. It reports the following in `decode.log`:
   - WER, with substitution, deletion, and insertion rates
   - the hallucination rates HR<sub>norm</sub> (non-empty output after text normalization) and HR<sub>raw</sub> (non-empty raw output)
   - disfluency F1

   To ablate SinkGate features at inference time, pass `--sink_gate_disable_features` with one or more of `max_prob`, `norm_entropy`, `shift`.

## Trained Models

AURA fine-tuned Whisper models are released on the [Hugging Face Hub](https://huggingface.co/balaji1312). Each checkpoint bundles the frozen Whisper backbone with the trained AURA parameters, and loads with the code in this repository. For example, from the repository root:

```bash
export rootdir=$(pwd)   # the scripts import src/ from $rootdir; run_whisper.sh sets this for you
huggingface-cli download balaji1312/whisper-medium-myst-aura --local-dir models/whisper-medium-myst-aura
mkdir -p results/myst_test
python src/bin/decode_asr.py \
  --wav_scp egs/MyST/data/test/wav.scp \
  --trn_scp egs/MyST/data/test/text \
  --model models/whisper-medium-myst-aura \
  --processor models/whisper-medium-myst-aura \
  --result_ref_file results/myst_test/ref.txt \
  --result_hyp_file results/myst_test/hyp.txt
```

|                         Model                          |                                Hugging Face                                |
| :----------------------------------------------------: | :------------------------------------------------------------------------: |
|        Whisper medium.en — MyST (child speech)         |    [model](https://huggingface.co/balaji1312/whisper-medium-myst-aura)     |
|    Whisper large-v3 — TED-LIUM 3 (imperfect labels)    |  [model](https://huggingface.co/balaji1312/whisper-largev3-tedlium-aura)   |
|   Whisper medium.en — FluencyBank (disfluent speech)   | [model](https://huggingface.co/balaji1312/whisper-medium-fluencybank-aura) |
| Whisper large-v3 — non-speech hallucination (5 epochs) | [model](https://huggingface.co/balaji1312/whisper-largev3-nonspeech-aura)  |

## Citation

If you use this code in your research, please cite it as follows:

```bibtex
@inproceedings{shankar2026aura,
  author    = {Shankar, Natarajan Balaji and Wang, Zilai and Wang, Zihan and Shi, Mohan and Zhang, Kaiyuan and Alwan, Abeer},
  title     = {{AURA}: Uncertainty-Routed Activation Editing for Acoustic Grounding in Speech Foundation Models},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026},
}
```
