#!/usr/bin/env bash
# List every trained version with its date, settings and artefacts.
#
#   bash deploy/taipei1/runs.sh              # table
#   bash deploy/taipei1/runs.sh -v           # plus the full run_config.json
#
# Runs on the login node; reads only.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

VERBOSE=${1:-}

mapfile -t dirs < <(list_runs_paths)
if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "no version directories under $ROOT (expected ${VERSION_GLOB})"
    exit 0
fi

printf '%-26s %-10s %-6s %-7s %-5s %-16s %s\n' \
    VERSION DATE EPOCHS LR GPUS ARTEFACTS 'EXTRA [CORPORA]'
printf '%.0s-' {1..126}; echo

for d in "${dirs[@]}"; do
    name=$(basename "$d")
    cfg="$d/run_config.json"
    # Count with globs, not `ls`: env.sh sets `pipefail`, and an unexpanded glob
    # makes `ls` exit non-zero, which under `set -e` aborted the whole listing the
    # moment it reached a version that had no eval/ directory yet.
    shopt -s nullglob
    ckpt_files=("$d"/model_step*.pth)
    eval_files=("$d"/eval/*.wav "$d"/eval/*/*.wav)
    shopt -u nullglob
    ckpts=${#ckpt_files[@]}
    evals=${#eval_files[@]}
    pruned=$([[ -f "$d/pruned/gpt.pth" ]] && echo " pruned" || echo "")
    evaltag=$([[ "$evals" -gt 0 ]] && echo " eval:$evals" || echo "")

    if [[ -f "$cfg" ]]; then
        read -r started epochs lr gpus extra corpora < <(
            python3 - "$cfg" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
h = c.get("hyperparams", {})
print(
    (c.get("started") or "?")[:10],
    h.get("EPOCHS") or "?",
    h.get("LR") or "?",
    c.get("num_gpus") or "?",
    # Spaces would be split into separate read targets, and long strings wreck
    # the column, so collapse and truncate.
    ((h.get("EXTRA_ARGS") or "-").replace(" ", ",")[:44] or "-"),
    c.get("corpora") or "?",
)
PY
        )
    else
        started="(no config)"; epochs="?"; lr="?"; gpus="?"; extra="-"; corpora="?"
    fi

    printf '%-26s %-10s %-6s %-7s %-5s %-16s %s\n' \
        "$name" "$started" "$epochs" "$lr" "$gpus" "ckpt:${ckpts}${pruned}${evaltag}" "$extra [$corpora]"

    if [[ "$VERBOSE" == "-v" && -f "$cfg" ]]; then
        sed 's/^/      /' "$cfg"
        echo
    fi
done

echo
echo "latest: $(latest_run)"
