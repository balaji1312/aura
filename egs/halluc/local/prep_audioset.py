#!/usr/bin/env python3
"""Export non-speech AudioSet clips (balanced train) as wav.scp/text with empty transcripts."""
import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

SPEECH_MID = "/m/09x0r"  # AudioSet "Speech"


def safe_id(s):
    s = (s or "").strip().replace("-", "_")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_") or "unk"


def is_speech(ex, strict=False):
    if SPEECH_MID in (ex.get("labels") or []):
        return True
    return strict and "speech" in " ".join(str(x).lower() for x in (ex.get("human_labels") or []))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", default=None, help="Hugging Face datasets cache dir")
    ap.add_argument("--out_dir", default="data/audioset/train")
    ap.add_argument("--num_samples", type=int, default=11753)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--strict_speech_filter", action="store_true",
                    help="also drop clips whose human-readable labels mention speech")
    args = ap.parse_args()

    out = Path(args.out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("agkphysics/AudioSet", name="balanced", split="train", cache_dir=args.cache_dir)
    ds = ds.cast_column("audio", Audio())
    ds = ds.filter(lambda ex: not is_speech(ex, strict=args.strict_speech_filter))
    ds = ds.shuffle(seed=args.seed)
    ds = ds.select(range(min(args.num_samples, len(ds))))

    scp, text = [], []
    for i, ex in enumerate(tqdm(ds, desc="AudioSet")):
        arr, sr = ex["audio"]["array"], int(ex["audio"]["sampling_rate"])
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            arr = arr.mean(axis=1)
        utt = f"audioset_baltrain_{i:07d}_{safe_id(ex.get('video_id', 'novid'))}"
        wav = wav_dir / f"{utt}.wav"
        sf.write(wav, arr, sr, subtype="PCM_16")
        scp.append(f"{utt} {wav.resolve()}")
        text.append(utt)

    (out / "wav.scp").write_text("\n".join(scp) + "\n")
    (out / "text").write_text("\n".join(text) + "\n")
    print(f"{len(scp)} clips -> {out}")


if __name__ == "__main__":
    main()
