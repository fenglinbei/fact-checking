#!/usr/bin/env bash
set -euo pipefail

latexmk -xelatex -interaction=nonstopmode -halt-on-error fact_checking_survey.tex
