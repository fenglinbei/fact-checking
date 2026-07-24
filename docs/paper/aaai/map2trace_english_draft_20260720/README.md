# Map2Trace: AAAI 2027 English Draft

This directory contains a faithful English migration of the supplied Chinese draft into the official AAAI 2027 anonymous-submission format. The manuscript remains intentionally uncompressed at this stage; content has not been removed to meet the page limit.

## Project Contents

- `manuscript.tex`: single-file English manuscript in anonymous-submission mode.
- `references.bib`: bibliography copied from the supplied Chinese project.
- `aaai2027.sty` and `aaai2027.bst`: unmodified files from the supplied official AAAI 2027 Author Kit.
- `figures/AAAI-FC-main.pdf`: the English framework figure used by the manuscript.
- `official/ReproducibilityChecklist.tex`: unmodified official checklist template to complete and upload separately.
- `build.sh`: PDFLaTeX/BibTeX build wrapper.
- `check_compliance.sh`: source, log, and PDF sanity checks.
- `dist/manuscript.pdf`: compiled draft PDF after a successful build.

## Build

Use a complete TeX Live or MiKTeX installation that provides the official `newtxtext` package required by `aaai2027.sty`.

```bash
./build.sh
./check_compliance.sh
```

The build uses PDFLaTeX, not XeLaTeX or LuaLaTeX. Generated files are isolated under `build/latex/` and `dist/`.

## AAAI 2027 Migration Decisions

- Uses `\documentclass[letterpaper]{article}` and `\usepackage[submission]{aaai2027}`.
- Uses `Anonymous Submission` as the sole author and leaves affiliations empty.
- Preserves the official `TemplateVersion (2027.1)` PDF information block.
- Keeps all manuscript text in one `.tex` file and uses the official BibTeX style through `aaai2027.sty`.
- Removes the Chinese-only XeLaTeX, CJK, font, and course-style dependencies.
- Uses no forbidden layout, hyperlink, font, page-break, or compression packages/commands.
- Places figure and table captions below their contents and keeps table text at the permitted 9-point size.
- Keeps the anonymous artifact URL in the template-provided `links` environment between the abstract and main text.
- Preserves all sections, equations, tables, citations, numerical results, placeholders, and stated evidence boundaries from the supplied draft.

The current AAAI-27 main-track instructions allow up to seven pages of technical content, with later pages reserved for references, and require the reproducibility checklist to be uploaded separately. This draft deliberately does not optimize for that limit yet. See the [AAAI-27 submission instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/) and [main-track call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/).

The current compiled draft is 12 pages including references.

## Deliberately Preserved Draft Placeholders

- `ANONYMIZED_URL` must be replaced with a genuinely anonymous repository URL before submission.
- Exp2 remains explicitly marked as a placeholder, including all `TBD` entries. No reliability values were invented.
- The official reproducibility checklist is still blank and must be completed separately before submission.
- Scientific claims and evidence-scope tensions already present in the Chinese draft were translated rather than silently rewritten. In particular, the current completed human study directly audits claim atomization, while the Evidence Map reliability study remains unfinished.

## Provenance

- Chinese project archive SHA-256: `7026fe1b52ab6874052358f35e8a71f246e686bc74883c30af950dc0fe85ac16`
- AAAI 2027 Author Kit archive SHA-256: `e28c6ac9bc6eb3b4e2d849547d2cefb5162610ee39d0a12e0dc62d1126b44a7d`
- Translation source: the supplied archive `AAAI_FC_202507200128.zip`; older repository drafts were not used as translation sources.
