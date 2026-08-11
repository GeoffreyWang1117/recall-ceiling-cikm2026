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

# ── Results (JSON only; no raw logs, no per-user API checkpoints) ─────────────
mkdir -p "$DEST/experiments/logs"
cp "$SRC"/experiments/logs/*.json "$DEST/experiments/logs/" 2>/dev/null || true

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
cp "$SRC/requirements.txt" "$DEST/" 2>/dev/null || true
cp "$SRC/config.yaml"      "$DEST/" 2>/dev/null || true
cp "$SRC/.env.example"     "$DEST/" 2>/dev/null || true

# ── Safety: never copy these ─────────────────────────────────────────────────
rm -f "$DEST/.env" "$DEST"/**/.env 2>/dev/null || true
rm -rf "$DEST/experiments/logs/checkpoints" "$DEST/.git" 2>/dev/null || true

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
experiments/logs/checkpoints/
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
