#!/usr/bin/env bash
# Build the public artifact repository for the CIKM 2026 camera-ready.
#
# Produces a self-contained staging tree with a FRESH git history, so no blob
# from the development repository (which contains a .env with live API keys) can
# ever be reachable from the public remote.
#
# Usage:  bash scripts/build_artifact_repo.sh [STAGING_DIR]
# Then:   inspect STAGING_DIR, run the secret scan it prints, and only then push.

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/tmp/claude-1000/-home-coder-gw-Projects-GraphLLMRec/0d8bb6fa-f733-437e-aacb-6b4e740a22f6/scratchpad/artifact-repo}"

echo "Source : $SRC"
echo "Staging: $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"

# ── Code ─────────────────────────────────────────────────────────────────────
mkdir -p "$DEST/scripts"
cp "$SRC"/scripts/*.py "$DEST/scripts/"
cp "$SRC"/scripts/*.sh "$DEST/scripts/" 2>/dev/null || true
for d in evaluation models utils analysis text_ablation; do
  [ -d "$SRC/$d" ] && cp -r "$SRC/$d" "$DEST/$d"
done
find "$DEST" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name "*.pyc" -delete 2>/dev/null || true
# Follow-up research code whose outputs are not part of this artifact.
# cikm_arm_decomposition.py is kept: README §5(a) invokes its --mode bound to
# recompute the Corollary 3 bounds.
rm -f "$DEST/scripts/cikm_arm_selection_analysis.py"

# ── Results (JSON only; no raw logs) ──────────────────────────────────────────
mkdir -p "$DEST/experiments/logs"
cp "$SRC"/experiments/logs/*.json "$DEST/experiments/logs/" 2>/dev/null || true
# Follow-up research outputs (post-camera-ready arm decomposition, smoke tests and
# partial analyses) are not part of the CIKM artifact.
rm -f "$DEST"/experiments/logs/cikm_arm_decomposition_*.json \
      "$DEST"/experiments/logs/cikm_arm_selection_*.json \
      "$DEST"/experiments/logs/partial_*.json \
      "$DEST"/experiments/logs/smoke_*.json

# ── Processed datasets actually used by the paper ─────────────────────────────
mkdir -p "$DEST/data/processed"
for d in amazon_beauty_sampled amazon_movies_sampled amazon_electronics_sampled \
         amazon_sports_sampled amazon_toys_sampled amazon_office_sampled \
         mind_news movielens_25m; do
  [ -d "$SRC/data/processed/$d" ] && cp -r "$SRC/data/processed/$d" "$DEST/data/processed/$d"
done

# ── Config / docs ─────────────────────────────────────────────────────────────
cp "$SRC/artifact/README.md" "$DEST/README.md"
cp "$SRC/artifact/LICENSE"   "$DEST/LICENSE"

# GitHub Pages project page, served from main:/docs. Must be staged here or a
# rebuild-and-force-push would delete the live page at
# https://geoffreywang1117.github.io/recall-ceiling-cikm2026/
[ -d "$SRC/artifact/docs" ] && cp -r "$SRC/artifact/docs" "$DEST/docs"
cp "$SRC/requirements.txt" "$DEST/" 2>/dev/null || true
cp "$SRC/config.yaml"      "$DEST/" 2>/dev/null || true
cp "$SRC/.env.example"     "$DEST/" 2>/dev/null || true

# ── Safety: never copy these ─────────────────────────────────────────────────
rm -f "$DEST/.env" "$DEST"/**/.env 2>/dev/null || true
rm -rf "$DEST/experiments/logs/checkpoints" "$DEST/.git" 2>/dev/null || true

# ── Per-user checkpoints: only files verified by scripts/verify_released_per_user.py ──
# The whitelist is the manifest that script writes; regenerate it before building.
MANIFEST="$SRC/experiments/logs/per_user_manifest.json"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST: run scripts/verify_released_per_user.py first" >&2; exit 1; }
mkdir -p "$DEST/experiments/logs/checkpoints"
python3 - "$MANIFEST" "$SRC/experiments/logs/checkpoints" "$DEST/experiments/logs/checkpoints" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
manifest, src, dst = map(Path, sys.argv[1:4])
files = [f["file"] for f in json.loads(manifest.read_text())["files"]]
missing = [f for f in files if not (src / f).is_file()]
if missing:
    sys.exit(f"manifest lists files that do not exist: {missing}")
for f in files:
    shutil.copy2(src / f, dst / f)
print(f"per-user checkpoints staged: {len(files)} files")
PYEOF

cat > "$DEST/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
.env
.env.local
*.pt
*.pth
*.bin
*.safetensors
.ipynb_checkpoints
.vscode/
.idea/
wandb/
EOF

echo
echo "=== staged size ==="
du -sh "$DEST"
echo
echo "=== secret scan (must print nothing) ==="
grep -rInE "sk-[A-Za-z0-9]{16,}|gsk_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xoxb-|-----BEGIN [A-Z ]*PRIVATE KEY" "$DEST" 2>/dev/null || true
echo "=== .env present? (must print nothing) ==="
find "$DEST" -name ".env" 2>/dev/null || true
echo
echo "Staged. Review, then: cd $DEST && git init && git add -A && git commit"
