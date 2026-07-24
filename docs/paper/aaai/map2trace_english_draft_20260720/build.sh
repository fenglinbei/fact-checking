#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_dir}/build/latex"
dist_dir="${project_dir}/dist"

mkdir -p "${build_dir}" "${dist_dir}"

if ! kpsewhich newtxtext.sty >/dev/null 2>&1; then
  echo "ERROR: newtxtext.sty is missing. Install the official newtx TeX package before building." >&2
  exit 1
fi

cd "${project_dir}"
latexmk \
  -pdf \
  -bibtex \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  -outdir="${build_dir}" \
  manuscript.tex

cp "${build_dir}/manuscript.pdf" "${dist_dir}/manuscript.pdf"

echo "Built ${dist_dir}/manuscript.pdf"
