#!/usr/bin/env bash
# Download the sky-facing subset of FRED (11 test + 44 train sequences): RGB frames and boxes only.
# Test sequences keep every frame, train sequences every 10th. Finished sequences are skipped.
#
#   bash scripts/fetch_fred.sh [out_dir]      # default: data/raw/fred
set -euo pipefail

BASE=https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main
OUT="${1:-data/raw/fred}"
TEST_IDS="12 60 63 64 65 80 93 110 111 115 219"
TRAIN_IDS="11 52 53 54 55 56 57 58 59 61 62 66 67 68 69 70 71 72 73 74 75 76 77 78 79 91 92 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 218"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT/train" "$OUT/test"

fetch() {  # split id stride
  local split=$1 id=$2 stride=$3 zip="$TMP/$2.zip"
  [ -f "$OUT/$split/$id/.done" ] && return 0
  curl -fsSL --retry 5 --retry-delay 10 -C - -o "$zip" "$BASE/$split/$id.zip"
  unzip -tqq "$zip"
  { unzip -Z1 "$zip" | grep -E '/RGB/[^/]+\.(jpg|png)$' | sort | awk -v s="$stride" 's == 1 || NR % s == 1'
    unzip -Z1 "$zip" | grep -E '/coordinates(_rgb)?\.txt$'; } > "$TMP/list"
  (cd "$OUT/$split" && tr '\n' '\0' < "$TMP/list" | xargs -0 unzip -qq -o "$zip")
  echo "stride=$stride" > "$OUT/$split/$id/.done"
  rm -f "$zip"
  echo "$split/$id"
}

for id in $TEST_IDS; do fetch test "$id" 1; done
for id in $TRAIN_IDS; do fetch train "$id" 10; done
