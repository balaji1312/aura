#!/usr/bin/env python3
"""Cut DEMAND 16 kHz recordings into chunks of at most 30 s with empty transcripts."""
import argparse
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

MAX_SEC = 30.0
MIN_SEC = 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demand_root", default="corpora/demand_16k")
    ap.add_argument("--out_dir", default="data/demand/train")
    args = ap.parse_args()

    out = Path(args.out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    scp, text, idx = [], [], 0
    for path in tqdm(sorted(Path(args.demand_root).rglob("*.wav")), desc="DEMAND"):
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            continue
        audio = audio.mean(axis=1)
        if len(audio) / sr < MIN_SEC:
            continue
        max_len = int(MAX_SEC * sr)
        for c in range((len(audio) + max_len - 1) // max_len):
            chunk = audio[c * max_len:(c + 1) * max_len]
            if len(chunk) / sr < MIN_SEC:
                continue
            utt = f"demand_{path.parent.name}_{path.stem}_{idx:09d}_c{c:03d}"
            wav = wav_dir / f"{utt}.wav"
            sf.write(wav, chunk, sr)
            scp.append(f"{utt} {wav.resolve()}")
            text.append(utt)
            idx += 1

    (out / "wav.scp").write_text("\n".join(scp) + "\n")
    (out / "text").write_text("\n".join(text) + "\n")
    print(f"{len(scp)} chunks -> {out}")


if __name__ == "__main__":
    main()
