#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_dir}/build/supplement"
dist_dir="${project_dir}/dist"

mkdir -p "${build_dir}" "${dist_dir}"

cd "${project_dir}"
latexmk \
  -pdf \
  -bibtex \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  -outdir="${build_dir}" \
  supplementary_material_draft.tex

cp \
  "${build_dir}/supplementary_material_draft.pdf" \
  "${dist_dir}/supplementary_material_draft.pdf"

echo "Built ${dist_dir}/supplementary_material_draft.pdf"
