#!/usr/bin/env bash
# Run all the experiments, or selected stages. Each stage skips work whose outputs exist,
# so the script can be re-run after an interruption. Logs go to runs/logs/.
#
#   bash scripts/run_experiments.sh                    # data prepare train estimate evaluate benchmark
#   bash scripts/run_experiments.sh train evaluate     # selected stages
#   bash scripts/run_experiments.sh --smoke            # every stage on tiny subsets (minutes)
set -o pipefail

if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]]; then
  CAFFEINATED=1 exec caffeinate -is bash "$0" "$@"   # no idle or system sleep while running
fi

smoke=() stages=() passed=() failed=()
for arg in "$@"; do
  case "$arg" in
    --smoke) smoke=(--smoke) ;;
    *) stages+=("$arg") ;;
  esac
done
(( ${#stages[@]} )) || stages=(data prepare train estimate evaluate benchmark)

logs=runs/logs
(( ${#smoke[@]} )) && logs=runs/smoke/logs
mkdir -p "$logs"
models=(yolo26s-mixed yolo26s-cenek)

step() {
  local name=$1; shift
  echo "==> $name"
  if uv run dronetrack "${smoke[@]}" "$@" 2>&1 | tee "$logs/$name.log"; then
    passed+=("$name")
  else
    failed+=("$name")
  fi
}

for stage in "${stages[@]}"; do
  case "$stage" in
    data)      step data data ;;
    prepare)   for m in "${models[@]}"; do step "prepare-$m" prepare "$m"; done ;;
    train)     for m in "${models[@]}"; do step "train-$m" train "$m"; step "export-$m" export "$m"; done ;;
    estimate)  step estimate-motion estimate motion
               for m in "${models[@]}"; do step "estimate-$m" estimate "$m"; done ;;
    evaluate)  for s in comparison ablation robustness realtime detection; do step "evaluate-$s" evaluate "$s"; done ;;
    benchmark) step benchmark benchmark ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done

echo
echo "passed: ${passed[*]:-none}"
echo "failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
