#!/usr/bin/env python3
"""Export UrbanSound8K as a non-speech evaluation set (wav.scp/text with empty references)."""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", default=None, help="Hugging Face datasets cache dir")
    ap.add_argument("--out_dir", default="data/urbansound8k")
    args = ap.parse_args()

    out = Path(args.out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("mteb/urbansound8K", cache_dir=args.cache_dir)
    split = ds[list(ds.keys())[0]].cast_column("audio", Audio())

    scp, text = [], []
    for i, ex in enumerate(tqdm(split, desc="UrbanSound8K")):
        arr, sr = ex["audio"]["array"], int(ex["audio"]["sampling_rate"])
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            arr = arr.mean(axis=1)
        utt = f"urb8k_{i:07d}"
        wav = wav_dir / f"{utt}.wav"
        sf.write(wav, arr, sr)
        scp.append(f"{utt} {wav.resolve()}")
        text.append(utt)

    (out / "wav.scp").write_text("\n".join(scp) + "\n")
    (out / "text").write_text("\n".join(text) + "\n")
    print(f"{len(scp)} clips -> {out}")


if __name__ == "__main__":
    main()
