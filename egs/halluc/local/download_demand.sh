#!/usr/bin/env bash
# Download the 16 kHz DEMAND noise recordings from Zenodo (record 1227121) and extract them.
set -euo pipefail

rec_id=${1:-1227121}
zip_dir=${2:-corpora/demand_zips}
out_dir=${3:-corpora/demand_16k}

mkdir -p "$zip_dir" "$out_dir"

python3 - "$rec_id" <<'EOF' |
import json, sys, urllib.request
rec = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{sys.argv[1]}"))
for f in rec["files"]:
    if f["key"].endswith("_16k.zip"):
        print(f["links"]["self"], f["key"], f["checksum"].split(":")[-1], sep="\t")
EOF
  while IFS=$'\t' read -r url key md5; do
    out="$zip_dir/$key"
    if [ ! -f "$out" ]; then
      curl -fL --retry 5 --retry-delay 2 -o "$out.part" "$url"
      mv "$out.part" "$out"
    fi
    [ "$(md5sum "$out" | awk '{print $1}')" = "$md5" ] || { echo "MD5 mismatch: $key" >&2; exit 1; }
  done

for z in "$zip_dir"/*_16k.zip; do
  unzip -q -o "$z" -d "$out_dir"
done
