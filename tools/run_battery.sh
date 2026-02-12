#!/bin/bash
# IONIS V20 Link Budget Validation Battery
# 23 profiles x 3 sources = 69 scoring runs
#
# Usage:
#   bash tools/run_battery.sh 2>&1 | tee battery_$(date +%Y%m%d_%H%M%S).log
#
# Monitor progress:
#   tail -f results/battery_progress.txt
#   watch -n 5 cat results/battery_progress.txt

set -euo pipefail

CONFIG="versions/v20/config_v20.json"
PYTHON="${PYTHON:-python}"
SCORER="tools/score_model.py"
PROGRESS_FILE="results/battery_progress.txt"

# Ensure results directory exists
mkdir -p results

PROFILES=(
    wspr wspr_dipole voacap_default
    qrp_milliwatt qrp_portable qrp_home sota_activator pota_activator
    home_vertical home_station home_beam
    home_amp_dipole home_amp_beam big_gun
    contest_lp contest_cw contest_ssb contest_super
    dxpedition_lite dxpedition dxpedition_mega
    maritime_mobile extreme_hf
)

SOURCES=(rbn pskr contest)

TOTAL_RUNS=$(( ${#PROFILES[@]} * ${#SOURCES[@]} ))
START_TIME=$(date +%s)

update_progress() {
    local run=$1
    local source=$2
    local profile=$3
    local status=$4
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local pct=$((run * 100 / TOTAL_RUNS))

    if [ $run -gt 0 ]; then
        local avg_per_run=$((elapsed / run))
        local remaining=$((TOTAL_RUNS - run))
        local eta_secs=$((remaining * avg_per_run))
        local eta_mins=$((eta_secs / 60))
        local eta_hrs=$((eta_mins / 60))
        eta_mins=$((eta_mins % 60))
    else
        local eta_hrs="?"
        local eta_mins="?"
    fi

    cat > "$PROGRESS_FILE" << EOF
================================================================================
  IONIS V20 Battery Progress
  $(date -u '+%Y-%m-%d %H:%M:%S UTC')
================================================================================
  Run:      ${run} / ${TOTAL_RUNS} (${pct}%)
  Current:  ${source} x ${profile}
  Status:   ${status}
  Pass:     ${PASS}
  Fail:     ${FAIL}
  Elapsed:  $((elapsed / 60))m $((elapsed % 60))s
  ETA:      ${eta_hrs}h ${eta_mins}m remaining
================================================================================
EOF
}

echo "======================================================================"
echo "  IONIS V20 Link Budget Validation Battery"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Profiles: ${#PROFILES[@]}  Sources: ${#SOURCES[@]}"
echo "  Total runs: ${TOTAL_RUNS}"
echo ""
echo "  Monitor progress: tail -f ${PROGRESS_FILE}"
echo "======================================================================"

PASS=0
FAIL=0
RUN=0

for source in "${SOURCES[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Source: ${source}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for profile in "${PROFILES[@]}"; do
        RUN=$((RUN + 1))
        echo ""
        echo "── Run ${RUN}/${TOTAL_RUNS}: ${source} x ${profile} ──"

        update_progress $RUN "$source" "$profile" "RUNNING"

        if $PYTHON "$SCORER" --config "$CONFIG" --source "$source" --profile "$profile"; then
            PASS=$((PASS + 1))
            update_progress $RUN "$source" "$profile" "PASS"
        else
            FAIL=$((FAIL + 1))
            update_progress $RUN "$source" "$profile" "FAIL"
            echo "  *** FAILED: ${source} x ${profile} ***"
        fi
    done
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "======================================================================"
echo "  BATTERY COMPLETE — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Total: ${RUN}  Pass: ${PASS}  Fail: ${FAIL}"
echo "  Runtime: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m $((TOTAL_TIME % 60))s"
echo "======================================================================"

# Final progress update
cat > "$PROGRESS_FILE" << EOF
================================================================================
  IONIS V20 Battery COMPLETE
  $(date -u '+%Y-%m-%d %H:%M:%S UTC')
================================================================================
  Total:    ${RUN} / ${TOTAL_RUNS}
  Pass:     ${PASS}
  Fail:     ${FAIL}
  Runtime:  $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m $((TOTAL_TIME % 60))s
================================================================================
EOF
