#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_TMP_DIR="${TMPDIR:-/tmp}/doc_scan_tex"
BUILD_DIR="$PROJECT_DIR/build/latex"
DIST_DIR="$PROJECT_DIR/dist"
PDF_NAME="course_report_aaai2027.pdf"

mkdir -p "$TEX_TMP_DIR/texmf-var" "$TEX_TMP_DIR/texmf-config" "$BUILD_DIR" "$DIST_DIR"

cd "$PROJECT_DIR"
TEXMFVAR="$TEX_TMP_DIR/texmf-var" \
TEXMFCONFIG="$TEX_TMP_DIR/texmf-config" \
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir="$BUILD_DIR" course_report_aaai2027.tex

cp "$BUILD_DIR/$PDF_NAME" "$DIST_DIR/$PDF_NAME"
printf 'Wrote %s\n' "$DIST_DIR/$PDF_NAME"
