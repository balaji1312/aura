#!/usr/bin/env python3
"""Kaldi-style wav.scp/text for a LibriSpeech split, with flac files piped through `flac -c -d -s`."""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--librispeech_root", required=True, help="directory containing test-clean/ and test-other/")
    ap.add_argument("--split", required=True, help="e.g. test-clean")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    wav, text = {}, {}
    for trans in sorted((Path(args.librispeech_root) / args.split).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, words = line.split(maxsplit=1)
            wav[utt] = f"flac -c -d -s {(trans.parent / f'{utt}.flac').resolve()} |"
            text[utt] = words

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wav.scp").write_text("".join(f"{u} {wav[u]}\n" for u in sorted(wav)))
    (out / "text").write_text("".join(f"{u} {text[u]}\n" for u in sorted(text)))
    print(f"{args.split}: {len(wav)} utterances -> {out}")


if __name__ == "__main__":
    main()
