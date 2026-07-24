#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manuscript="${project_dir}/manuscript.tex"
log_file="${project_dir}/build/latex/manuscript.log"
pdf_file="${project_dir}/dist/manuscript.pdf"

if rg -n '[\p{Han}]' "${manuscript}" "${project_dir}/references.bib"; then
  echo "ERROR: Chinese characters remain in the submission source." >&2
  exit 1
fi

if rg -n '\\usepackage\{(xeCJK|CJK|hyperref|geometry|fullpage|titlesec|setspace|balance|flushend|stfloats|float|wrapfig|multicol|authblk|ulem|times|lmodern)\}' "${manuscript}"; then
  echo "ERROR: A package forbidden by the AAAI 2027 author kit is present." >&2
  exit 1
fi

if rg -n '\\(nocopyright|addtolength|baselinestretch|linespread|clearpage|newpage|pagebreak|pagestyle|tiny|scriptsize|resizebox)\b|\\v(space|skip)\{-' "${manuscript}"; then
  echo "ERROR: A command forbidden by the AAAI 2027 author kit is present." >&2
  exit 1
fi

if [[ ! -f "${log_file}" || ! -f "${pdf_file}" ]]; then
  echo "ERROR: Build artifacts are missing; run ./build.sh first." >&2
  exit 1
fi

if rg -n 'Overfull \\[hv]box|LaTeX Warning: (Citation|Reference).+undefined|There were undefined references' "${log_file}"; then
  echo "ERROR: The LaTeX log contains overflow or unresolved-reference warnings." >&2
  exit 1
fi

page_count="$(gs -q --permit-file-read="${pdf_file}" -dNODISPLAY \
  -c "(${pdf_file}) (r) file runpdfbegin pdfpagecount = quit")"
echo "Compliance checks passed; generated PDF pages: ${page_count}."
