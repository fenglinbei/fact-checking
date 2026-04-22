from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from fact_checking.utils.io import read_json, read_jsonl


def build_structured_input(
    claim: str,
    label: str,
    selected_subclaims: list[dict[str, Any]],
    support_evidence: list[dict[str, Any]],
    refute_evidence: list[dict[str, Any]],
) -> str:
    lines = [
        f"Claim: {claim}",
        f"Verdict: {label}",
        "",
        "Subclaims:",
    ]
    if selected_subclaims:
        for i, item in enumerate(selected_subclaims, start=1):
            lines.append(f"{i}. {item.get('text') or item.get('subclaim')}")
    else:
        lines.append("1. " + claim)

    lines.append("")
    lines.append("Support evidence:")
    if support_evidence:
        for x in support_evidence:
            lines.append(f"- {x.get('text') or x.get('sentence')}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Refute evidence:")
    if refute_evidence:
        for x in refute_evidence:
            lines.append(f"- {x.get('text') or x.get('sentence')}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Write a concise fact-checking explanation grounded only in the evidence.")
    return "\n".join(lines)


class ExplainerTrainDataset(Dataset):
    def __init__(self, raw_json_path: str | Path, graph_pred_path: str | Path) -> None:
        raw_items = read_json(raw_json_path)
        pred_items = read_jsonl(graph_pred_path)
        raw_by_id = {str(x["event_id"]): x for x in raw_items}

        self.records: list[dict[str, str]] = []
        for pred in pred_items:
            event_id = str(pred["event_id"])
            raw = raw_by_id.get(event_id)
            if raw is None:
                continue
            source = build_structured_input(
                claim=pred["claim"],
                label=str(raw.get("label") or pred.get("gold_label") or pred.get("pred_label")),
                selected_subclaims=pred.get("selected_subclaims", []),
                support_evidence=pred.get("support_evidence", []),
                refute_evidence=pred.get("refute_evidence", []),
            )
            target = str(raw.get("explain", "")).strip()
            if target:
                self.records.append({"source": source, "target": target, "event_id": event_id})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.records[idx]
