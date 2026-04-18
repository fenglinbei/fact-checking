from __future__ import annotations

from typing import Any


def _sent_text(x: dict[str, Any]) -> str:
    return str(x.get("text") or x.get("sentence") or x.get("sent") or "").strip()


class TemplateExplainer:
    def build(self, claim: str, pred_label: str, support_evidence: list[dict[str, Any]], refute_evidence: list[dict[str, Any]]) -> str:
        sup = [_sent_text(x) for x in support_evidence if _sent_text(x)]
        ref = [_sent_text(x) for x in refute_evidence if _sent_text(x)]

        parts = [f'The claim is assessed as "{pred_label}".']
        if sup:
            parts.append("Supporting evidence indicates that " + sup[0].rstrip(".") + ".")
        if ref:
            parts.append("At the same time, counter-evidence suggests that " + ref[0].rstrip(".") + ".")
        if len(sup) > 1:
            parts.append("Additional support comes from " + sup[1].rstrip(".") + ".")
        if len(ref) > 1:
            parts.append("However, another limitation is that " + ref[1].rstrip(".") + ".")

        return " ".join(parts)
