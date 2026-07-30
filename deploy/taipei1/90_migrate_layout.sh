#!/usr/bin/env bash
# One-off: move the pre-versioning artefacts into per-version directories.
#
# Before, a version's pieces were scattered: checkpoints under runs/, the pruned
# weights and listening samples loose in outputs/ with inconsistent names
# (model_step23560_pruned.pth vs v2_step11780_pruned.pth, eval/ vs eval_v2/), and
# nothing tying either to a version. Afterwards each version is one directory.
#
# Idempotent, and it refuses to touch a run that a job is still writing to.
#
#   bash deploy/taipei1/90_migrate_layout.sh           # dry run
#   APPLY=1 bash deploy/taipei1/90_migrate_layout.sh   # do it

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

APPLY=${APPLY:-0}
run() {
    if [[ "$APPLY" == "1" ]]; then
        echo "  + $*"
        "$@"
    else
        echo "  would: $*"
    fi
}

# old run directory : new version directory : pruned checkpoint : eval directory
MAP=(
    "$ROOT/runs/tai8_v1|$ROOT/v1_20260729_191628|$ROOT/outputs/model_step23560_pruned.pth|$ROOT/outputs/eval"
    "$ROOT/runs/20260730_104546__tai8_v2_emotarget|$ROOT/v2_20260730_104546|$ROOT/outputs/v2_step11780_pruned.pth|$ROOT/outputs/eval_v2"
    "$ROOT/runs/20260730_154608__tai8_v3a_emodrop|$ROOT/v3a_20260730_154608||"
)

echo "=== jobs currently running (a live run must not be moved) ==="
squeue -h -u "$USER" -o '%i %j %T' 2>/dev/null || echo "  (squeue unavailable)"
echo

for entry in "${MAP[@]}"; do
    IFS='|' read -r old new pruned evaldir <<<"$entry"
    name=$(basename "$new")
    echo "--- $name ---"

    if [[ -d "$new" && ! -d "$old" ]]; then
        echo "  already migrated"
    elif [[ ! -d "$old" ]]; then
        echo "  source missing ($old) — skipping"
        continue
    else
        # latest.pth is rewritten at every checkpoint, so a recent mtime means a
        # job is probably still writing here.
        if [[ -f "$old/latest.pth" ]]; then
            age=$(( $(date +%s) - $(stat -c %Y "$old/latest.pth") ))
            if [[ "$age" -lt 600 ]]; then
                echo "  latest.pth changed ${age}s ago — looks live, skipping"
                continue
            fi
        fi
        run mv "$old" "$new"
    fi

    if [[ -n "$pruned" && -f "$pruned" ]]; then
        run mkdir -p "$new/pruned"
        run mv "$pruned" "$new/pruned/gpt.pth"
        [[ "$APPLY" == "1" ]] && basename "$pruned" >"$new/pruned/SOURCE"
    fi

    if [[ -n "$evaldir" && -d "$evaldir" ]]; then
        run mkdir -p "$new/eval"
        for f in "$evaldir"/*; do
            [[ -e "$f" ]] && run mv "$f" "$new/eval/"
        done
        run rmdir "$evaldir"
    fi

    # Point the version at its own Slurm log.
    if [[ -f "$new/run_config.json" ]]; then
        job=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))\
.get('slurm_job_id') or '')" "$new/run_config.json" 2>/dev/null || true)
        if [[ -n "$job" && -f "$ROOT/logs/train_${job}.out" && ! -e "$new/train.log" ]]; then
            run ln -s "../logs/train_${job}.out" "$new/train.log"
        fi
    fi
done

echo
echo "--- leftovers in outputs/ ---"
ls -la "$ROOT/outputs/" 2>/dev/null | tail -n +2
echo
echo "(v1 has no run_config.json: it predates the recorded-config change. Its"
echo " settings are in report/TRAINING_LOG.md — 10 epochs, LR 1e-5,"
echo " emotion-source=prompt, job 175964.)"

if [[ "$APPLY" != "1" ]]; then
    echo
    echo "dry run — set APPLY=1 to perform the moves"
fi
