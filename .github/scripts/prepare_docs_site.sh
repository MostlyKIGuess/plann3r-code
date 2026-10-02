#!/usr/bin/env bash
# Prepare docs/ for the MkDocs site: copy the README files in as pages and turn
# links that leave docs/ into GitHub links. Run from the repository root before
# `mkdocs build`. The generated pages are listed in .gitignore.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/MostlyKIGuess/plann3r-code}"
BRANCH="${BRANCH:-main}"
BLOB="$REPO_URL/blob/$BRANCH"

# README.md links to docs/<page>.md, which is <page>.md inside the site.
# Other relative links point at repository files.
sed -E \
  -e 's#\]\(docs/#](#g' \
  -e "s#\]\(real_world/README\.md\)#](real-world-code.md)#g" \
  -e "s#\]\(history/README\.md\)#](history.md)#g" \
  -e "s#\]\((\./)?([A-Za-z_][A-Za-z0-9_./-]*)\)#](${BLOB}/\2)#g" \
  README.md > docs/index.md
# The last rule also matched the rewritten page links. Restore them.
sed -i -E "s#\]\(${BLOB}/([a-z-]+\.md(\#[^)]*)?)\)#](\1)#g" docs/index.md

# real_world/README.md links to ../docs/<page>.md and to files beside it.
sed -E \
  -e 's#\]\(\.\./docs/#](#g' \
  -e "s#\]\((\./)?([A-Za-z_][A-Za-z0-9_./-]*)\)#](${BLOB}/real_world/\2)#g" \
  real_world/README.md > docs/real-world-code.md
sed -i -E "s#\]\(${BLOB}/real_world/([a-z-]+\.md(\#[^)]*)?)\)#](\1)#g" docs/real-world-code.md

sed -E \
  -e 's#\]\(\.\./docs/#](#g' \
  -e "s#\]\((\./)?([A-Za-z_][A-Za-z0-9_./-]*)\)#](${BLOB}/history/\2)#g" \
  history/README.md > docs/history.md
sed -i -E "s#\]\(${BLOB}/history/([a-z-]+\.md(\#[^)]*)?)\)#](\1)#g" docs/history.md

# Pages in docs/ that link up into the repository (../real_world and so on).
for page in docs/*.md; do
  sed -i -E "s#\]\(\.\./([^)]*)\)#](${BLOB}/\1)#g" "$page"
done
