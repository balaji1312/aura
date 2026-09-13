#!/usr/bin/env python3
"""Merge the MUSAN, DEMAND and AudioSet splits into one shuffled non-speech training set."""
import argparse
import random
from pathlib import Path


def read_scp(path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        utt, rest = line.split(maxsplit=1)
        if utt in out:
            raise ValueError(f"duplicate utt_id in {path}: {utt}")
        out[utt] = rest
    return out


def read_text(path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        out[parts[0]] = parts[1] if len(parts) == 2 else ""
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="data/train")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pairs", nargs="+", default=["data/musan/train", "data/demand/train", "data/audioset/train"])
    args = ap.parse_args()

    items, seen = [], set()
    for d in map(Path, args.pairs):
        wav, txt = read_scp(d / "wav.scp"), read_text(d / "text")
        if set(wav) != set(txt):
            raise ValueError(f"{d}: wav.scp and text list different utterances")
        for utt, path in wav.items():
            if utt in seen:
                raise ValueError(f"duplicate utt_id across sets: {utt}")
            seen.add(utt)
            items.append((utt, path, txt[utt]))

    random.Random(args.seed).shuffle(items)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wav.scp").write_text("\n".join(f"{u} {p}" for u, p, _ in items) + "\n")
    (out / "text").write_text("\n".join(f"{u} {t}".rstrip() for u, _, t in items) + "\n")
    print(f"{len(items)} clips -> {out}")


if __name__ == "__main__":
    main()
