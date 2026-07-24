# Structure-Only AAAI-Style Manuscript v0.4.2

This is a self-contained XeLaTeX migration of
`../aaai/writing_outline_v0.4.2_structure_only.md`. It uses the repository's
Chinese-capable `aaai2027-course.sty`, which reproduces the AAAI visual style
but is **not** the official AAAI submission template.

## Build

From the repository root:

```bash
bash docs/paper/aaai_structure_only_v0_4_2_pdf/build.sh
```

The generated PDF is written to:

```text
docs/paper/aaai_structure_only_v0_4_2_pdf/dist/manuscript.pdf
```

## Project files

- `manuscript.tex`: migrated paper body, tables, formulas, pseudocode, method
  figure, cross-references, and bibliography hook.
- `references.bib`: complete curated bibliography copied from
  `../aaai/references_structure_only.bib`.
- `figures/AAAI-FC-main-original.pdf`: byte-identical copy of the requested
  `../aaai/fig/AAAI-FC-main.pdf` source asset.
- `figures/AAAI-FC-main.pdf`: the 300-DPI compatibility rendering actually
  embedded by XeLaTeX, wrapped as PDF 1.5. The fallback is necessary because
  both the original tagged PDF 1.7 and a vector-only rewrap crash the server's
  XeLaTeX `xdvipdfmx` driver when embedded directly.
- `aaai2027-course.sty`: the existing XeLaTeX/CJK course-report style.
- `aaai2027.bst`: the bibliography style required by the course style.

## Draft status retained from the Markdown source

The migration intentionally leaves content-level placeholders unchanged:
`ANONYMIZED_URL`, the Exp2 reliability placeholder, and its `TBD` cells. The
working title and anonymous author block are centralized near the top of
`manuscript.tex` for later replacement.
