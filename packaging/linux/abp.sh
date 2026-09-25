#!/usr/bin/env bash
# ABP launcher for Linux: runs "python3 -m abp ...", keeps a UTF-8 launcher log
# in ${XDG_STATE_HOME:-~/.local/state}/abp/logs and returns ABP's exit status.
# Python interpreter: $ABP_PYTHON if set, otherwise python3.
set -u -o pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
log_dir="${XDG_STATE_HOME:-$HOME/.local/state}/abp/logs"
mkdir -p "$log_dir"
log="$log_dir/abp-$(date +%Y%m%d-%H%M%S-%N).log"
py="${ABP_PYTHON:-python3}"
{ echo "started: $(date -Is)"; printf 'command: %q ' "$py" -m abp "$@"; echo; } >> "$log"
if ! command -v "$py" >/dev/null 2>&1; then
  echo "error: Python interpreter '$py' not found (set ABP_PYTHON)" | tee -a "$log"
  echo "exit_code: 127" >> "$log"
  exit 127
fi
"$py" -m abp "$@" 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
{ echo "exit_code: $rc"; echo "finished: $(date -Is)"; } >> "$log"
[ "$rc" -ne 0 ] && echo "ABP exited with status $rc; launcher log: $log" >&2
exit "$rc"
