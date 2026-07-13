# SciFact Official Dev Comparison

## Reporting Scope

Use the original SciFact development split (300 claims) and the original
5,183-abstract corpus. The main table reports the four official full-pipeline
micro-F1 metrics in percentage points. It excludes oracle/gold-evidence,
evidence-provided classification, retrieval-only, SciFact-Open, and hidden-test
results.

This is a development-set comparison, not a hidden-test result. The paper should
state that the documented AI2 hidden-test leaderboard and the official contact
channel were unavailable when the experiment was finalized. Because this split
was also used for checkpoint selection, avoid claims of hidden-test SOTA.

## Recommended Main Table

**Table X: Full-pipeline performance on the official SciFact development split.**
All values are micro-F1 percentages. Bold marks the best result and underline
marks the second best result in each column.

| Method | Year | Sent. Selection-only | Sent. Selection+Label | Abstract Label-only | Abstract Label+Rationale |
|---|---:|---:|---:|---:|---:|
| VeriSci | 2020 | 48.30 | 43.10 | 52.10 | 50.00 |
| VerT5erini | 2021 | 60.87 | 57.10 | 65.07 | 61.72 |
| ParagraphJoint | 2021 | 64.70 | 55.20 | 65.10 | 59.90 |
| ARSJoint | 2021 | 66.20 | 57.80 | <u>66.70</u> | 62.40 |
| QMUL-SDS | 2021 | <u>67.83</u> | <u>60.54</u> | 63.40 | 61.10 |
| RerrFact | 2022 | **76.37** | **63.76** | 64.59 | <u>64.02</u> |
| PrunE | 2025 | 62.96 | 53.29 | 63.21 | 60.10 |
| **Atom-Union MREC (ours)** | -- | 40.51 | 39.23 | **72.41** | **65.82** |

The rows through RerrFact are the dev results collected in Table 4 of RerrFact.
PrunE is taken from its Table 2. PrunE uses the same official train/dev split and
corpus, but retrieves a top-150 bigram-TF-IDF universe and samples 12 candidate
abstracts for development inference; only the PrunE row is imported, not its
retrained baseline rows.

## Copy-ready LaTeX

```latex
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lccccc}
\toprule
& & \multicolumn{2}{c}{Sentence-level} & \multicolumn{2}{c}{Abstract-level} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Method & Year & Sel.-only & Sel.+Label & Label-only & Label+Rat. \\
\midrule
VeriSci          & 2020 & 48.30 & 43.10 & 52.10 & 50.00 \\
VerT5erini       & 2021 & 60.87 & 57.10 & 65.07 & 61.72 \\
ParagraphJoint   & 2021 & 64.70 & 55.20 & 65.10 & 59.90 \\
ARSJoint         & 2021 & 66.20 & 57.80 & \underline{66.70} & 62.40 \\
QMUL-SDS         & 2021 & \underline{67.83} & \underline{60.54} & 63.40 & 61.10 \\
RerrFact         & 2022 & \textbf{76.37} & \textbf{63.76} & 64.59 & \underline{64.02} \\
PrunE$^{\dagger}$ & 2025 & 62.96 & 53.29 & 63.21 & 60.10 \\
\midrule
Atom-Union MREC (ours) & -- & 40.51 & 39.23 & \textbf{72.41} & \textbf{65.82} \\
\bottomrule
\end{tabular}
\caption{Full-pipeline micro-F1 (\%) on the official SciFact development split
(300 claims) using the original 5,183-abstract corpus. Baselines through
RerrFact are reported by \citet{rana2022rerrfact}; PrunE is reported by
\citet{fang2025automatic}. Best results are bold and second-best results are
underlined. $^{\dagger}$PrunE uses a top-150 bigram-TF-IDF retrieval universe
and samples 12 candidate abstracts at development inference.}
\label{tab:scifact-dev-main}
\end{table*}
```

## Official-scorer Audit of Our Row

The exported validation predictions were rerun with the official SciFact
`verisci/evaluate/pipeline.py` and `verisci/evaluate/lib/metrics.py` at repository
commit `68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`. The official scorer and the
local exporter agree exactly:

| Metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| Sentence Selection-only | 38.16 | 43.17 | 40.51 |
| Sentence Selection+Label | 36.96 | 41.80 | 39.23 |
| Abstract Label-only | 76.88 | 68.42 | 72.41 |
| Abstract Label+Rationale | 69.89 | 62.20 | 65.82 |

Claim-level accuracy (82.00) and macro-F1 (81.35) are local diagnostics, not
official SciFact leaderboard metrics, and should not be added to this main
comparison table.

## Paper-safe Interpretation

The method obtains the strongest abstract-level result among the directly
comparable development-set systems: 72.41 Label-only F1 and 65.82
Label+Rationale F1. Relative to the second-best rows, these are improvements of
5.71 and 1.80 points, respectively. This supports a claim that the transferred
method preserves strong document-level scientific verification and can attach
at least one valid rationale to a correctly labeled abstract.

The same table also exposes a substantial sentence-localization weakness. The
method reaches 40.51 Selection-only F1 and 39.23 Selection+Label F1, trailing
RerrFact by 35.86 and 24.53 points. Therefore, do not describe the result as
overall SciFact SOTA. The defensible conclusion is strong abstract-level
transfer with weak exact sentence-level rationale recovery.

Suggested results text:

> Since the official SciFact hidden-test evaluation service was unavailable, we
> report full-pipeline results on the official development split. Atom-Union
> MREC achieves 72.41 abstract Label-only F1 and 65.82 abstract
> Label+Rationale F1, outperforming the strongest directly comparable dev
> baselines by 5.71 and 1.80 points, respectively. In contrast, sentence-level
> Selection+Label F1 remains 39.23, indicating that method-level transfer is
> substantially stronger for abstract verification than for exact rationale
> localization.

## Exclusions

- MultiVerS and BEVERS: original SciFact hidden-test results, not dev results in
  the same reporting table.
- Deka et al. (2023), CliVER, UNOWN, SYNTHVERIFY, and related LLM studies:
  evidence-provided or verification-only settings rather than open-corpus full
  pipelines.
- BEIR SciFact: retrieval-only metrics such as nDCG@10.
- SciFact-Open: a different approximately 500K-document retrieval benchmark.
- PrunE's KGAT, Paragraph-Joint, and ARSJoint reruns: reimplemented with a shared
  PubMedBERT backbone and PrunE-specific candidate protocol; retaining both
  these and the canonical rows would create duplicate, conflicting baselines.

## Sources

- SciFact official evaluation definition and format:
  https://github.com/allenai/scifact/blob/master/doc/evaluation.md
- Official scorer:
  https://github.com/allenai/scifact/blob/master/verisci/evaluate/pipeline.py
- RerrFact, Table 4:
  https://ceur-ws.org/Vol-3164/paper11.pdf
- PrunE, Table 2:
  https://openreview.net/attachment?id=cYAFwjY2bY&name=pdf
- Full precision/recall/F1 transcription:
  `docs/paper/aaai/scifact_dev_comparison_metrics.csv`
- Our official-scorer artifact:
  `outputs/sentence_trace_method/scifact__ministral3_8b__atom_union_fullpool_minmax9_9_lora_ebs16_lr2em5_ep12_eval100_pat8/submission/scifact_official_scorer_metrics_val.json`

## Minimal BibTeX Additions

```bibtex
@inproceedings{rana2022rerrfact,
  title={RerrFact: Reduced Evidence Retrieval Representations for Scientific Claim Verification},
  author={Rana, Ashish and Khanna, Deepanshu and Ghosal, Tirthankar and Singh, Muskaan and Singh, Harpreet and Rana, Prashant Singh},
  booktitle={Proceedings of the Second Workshop on Scientific Document Understanding},
  year={2022},
  url={https://ceur-ws.org/Vol-3164/paper11.pdf}
}

@inproceedings{fang2025automatic,
  title={Automatic Scientific Claims Verification with Pruned Evidence Graph},
  author={Fang, Liri and Fu, Dongqi and Torvik, Vetle I.},
  booktitle={ICLR 2025 Workshop on Agentic AI},
  year={2025},
  url={https://openreview.net/forum?id=cYAFwjY2bY}
}
```
