#!/usr/bin/env bash
# Install the vlm_usage_monitor package into the current Python environment and
# register this folder as a skill for Claude Code (and, if present, Codex).
#   bash install.sh            # pip install + skill symlinks
#   bash install.sh --no-pip   # skill symlinks only
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" != "--no-pip" ]]; then
  python -m pip install -e "$HERE"
fi

link_skill() {
  local dir="$1"
  mkdir -p "$dir"
  ln -sfn "$HERE" "$dir/vlm-usage-monitor"
  echo "Linked skill: $dir/vlm-usage-monitor -> $HERE"
}

link_skill "$HOME/.claude/skills"
[[ -d "$HOME/.codex" ]] && link_skill "$HOME/.codex/skills"
[[ -d "$HOME/.agents" ]] && link_skill "$HOME/.agents/skills"
echo "Done. Restart your agent session so it picks up the skill."
