from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

NODE_COLORS = {
    "claim": "#4C78A8",
    "subclaim": "#72B7B2",
    "support_evidence": "#54A24B",
    "refute_evidence": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize bad cases for Stage B / C(D).")
    parser.add_argument("--stage_b_predictions", type=str, default=None, help="Path to stage_b_predictions_*.jsonl")
    parser.add_argument("--graph_predictions", type=str, default=None, help="Path to *.graph_predictions.jsonl")
    parser.add_argument("--graph_inputs", type=str, default=None, help="Path to *.graph.jsonl from build_graph_inputs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save figures and csv")
    parser.add_argument("--top_n_badcases", type=int, default=50)
    parser.add_argument("--max_graph_cases", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_confusion_matrix(rows: list[dict[str, Any]], save_path: Path, title: str) -> list[list[int]]:
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0 for _ in LABELS] for _ in LABELS]

    for row in rows:
        gold = str(row.get("gold_label", row.get("label", ""))).strip().lower()
        pred = str(row.get("pred_label", "")).strip().lower()
        if gold not in label_to_id or pred not in label_to_id:
            continue
        matrix[label_to_id[gold]][label_to_id[pred]] += 1

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = matrix[i][j]
            if val > 0:
                plt.text(j, i, str(val), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return matrix


def rank_distance(gold_label: str, pred_label: str) -> int:
    order = {label: idx for idx, label in enumerate(LABELS)}
    if gold_label not in order or pred_label not in order:
        return -1
    return abs(order[gold_label] - order[pred_label])


def export_badcases_csv(rows: list[dict[str, Any]], out_csv: Path, top_n: int) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        gold = str(row.get("gold_label", row.get("label", ""))).strip().lower()
        pred = str(row.get("pred_label", "")).strip().lower()
        if not gold or not pred or gold == pred:
            continue
        support_num = len(row.get("support_evidence", []))
        refute_num = len(row.get("refute_evidence", []))
        enriched.append(
            {
                "event_id": row.get("event_id", ""),
                "claim": row.get("claim", ""),
                "gold_label": gold,
                "pred_label": pred,
                "rank_distance": rank_distance(gold, pred),
                "support_evidence_count": support_num,
                "refute_evidence_count": refute_num,
                "subclaim_count": len(row.get("selected_subclaims", [])),
            }
        )

    enriched.sort(key=lambda x: (x["rank_distance"], x["support_evidence_count"] + x["refute_evidence_count"]), reverse=True)
    top_rows = enriched[:top_n]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "claim",
                "gold_label",
                "pred_label",
                "rank_distance",
                "support_evidence_count",
                "refute_evidence_count",
                "subclaim_count",
            ],
        )
        writer.writeheader()
        writer.writerows(top_rows)

    return top_rows


def plot_error_distance(rows: list[dict[str, Any]], save_path: Path, title: str) -> None:
    counter: Counter[int] = Counter()
    for row in rows:
        gold = str(row.get("gold_label", row.get("label", ""))).strip().lower()
        pred = str(row.get("pred_label", "")).strip().lower()
        if gold == pred:
            continue
        dist = rank_distance(gold, pred)
        if dist >= 0:
            counter[dist] += 1

    xs = sorted(counter)
    ys = [counter[x] for x in xs]

    plt.figure(figsize=(6, 4))
    plt.bar(xs, ys, color="#E45756")
    plt.xlabel("Ordinal distance |gold - pred|")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def summarize_stage_c_gating(graph_items: list[dict[str, Any]], save_path: Path) -> None:
    reason_counter: Counter[str] = Counter()
    split_counter = Counter({"split": 0, "no_split": 0})
    subclaim_claim_sims: list[float] = []

    for item in graph_items:
        decomp = item.get("decomposition", {})
        gate = decomp.get("gate", {})
        should_split = bool(gate.get("should_split", False))
        split_counter["split" if should_split else "no_split"] += 1
        reason = str(gate.get("reason", "unknown"))
        reason_counter[reason] += 1
        for a in decomp.get("assignments", []):
            sim = a.get("subclaim_claim_similarity")
            if isinstance(sim, (int, float)):
                subclaim_claim_sims.append(float(sim))

    labels = ["split", "no_split"]
    values = [split_counter[k] for k in labels]

    plt.figure(figsize=(13, 4))
    plt.subplot(1, 3, 1)
    plt.bar(labels, values, color=["#54A24B", "#4C78A8"])
    plt.title("Stage C gating decisions")
    plt.ylabel("Count")

    top_reason = reason_counter.most_common(8)
    plt.subplot(1, 3, 2)
    plt.barh([x[0] for x in reversed(top_reason)], [x[1] for x in reversed(top_reason)], color="#72B7B2")
    plt.title("Top gating reasons")

    plt.subplot(1, 3, 3)
    if subclaim_claim_sims:
        plt.hist(subclaim_claim_sims, bins=20, color="#F58518", alpha=0.85)
        plt.axvline(sum(subclaim_claim_sims) / len(subclaim_claim_sims), color="#E45756", linestyle="--", linewidth=1.5)
    plt.title("Subclaim-claim similarity")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_subclaim_vs_error(pred_rows: list[dict[str, Any]], save_path: Path) -> None:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in pred_rows:
        gold = str(row.get("gold_label", row.get("label", ""))).strip().lower()
        pred = str(row.get("pred_label", "")).strip().lower()
        subclaims = len(row.get("selected_subclaims", []))
        d = rank_distance(gold, pred)
        if d >= 0:
            grouped[subclaims].append(d)

    xs = sorted(grouped)
    ys = [sum(grouped[k]) / max(1, len(grouped[k])) for k in xs]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", color="#F58518")
    plt.xlabel("#selected_subclaims")
    plt.ylabel("Avg ordinal error distance")
    plt.title("Graph prediction error vs subclaim count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_badcase_graphs(
    badcases: list[dict[str, Any]],
    graph_items: list[dict[str, Any]],
    out_dir: Path,
    max_cases: int,
) -> int:
    try:
        import networkx as nx
    except ImportError:
        print("[WARN] networkx not installed, skip graph topology figures.")
        return 0

    graph_by_id = {str(item.get("event_id")): item for item in graph_items}
    saved = 0

    for row in badcases:
        event_id = str(row.get("event_id"))
        if event_id not in graph_by_id:
            continue

        item = graph_by_id[event_id]
        nodes = item.get("nodes", [])
        edges = item.get("edges", [])
        if not nodes:
            continue

        g = nx.DiGraph()
        for node in nodes:
            node_id = int(node["node_id"])
            node_type = str(node.get("node_type", "unknown"))
            text = str(node.get("text", ""))
            g.add_node(node_id, node_type=node_type, label=text[:42] + ("..." if len(text) > 42 else ""))

        for edge in edges:
            g.add_edge(int(edge["src"]), int(edge["dst"]), rel=edge.get("rel_type", ""))

        pos = nx.spring_layout(g, seed=42)
        plt.figure(figsize=(9, 7))
        node_colors = [NODE_COLORS.get(g.nodes[n]["node_type"], "#B279A2") for n in g.nodes]

        nx.draw_networkx_nodes(g, pos, node_size=700, node_color=node_colors, alpha=0.95)
        nx.draw_networkx_labels(g, pos, labels={n: g.nodes[n]["label"] for n in g.nodes}, font_size=7)
        nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", width=1.0, alpha=0.45)

        plt.title(f"Bad case graph: {event_id}")
        plt.axis("off")
        out_path = out_dir / f"badcase_graph_{event_id.replace('/', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        saved += 1
        if saved >= max_cases:
            break

    return saved


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.stage_b_predictions and not args.graph_predictions:
        raise ValueError("At least one of --stage_b_predictions / --graph_predictions must be provided")

    if args.stage_b_predictions:
        stage_b_rows = read_jsonl(args.stage_b_predictions)
        plot_confusion_matrix(stage_b_rows, out_dir / "stage_b_confusion_matrix.png", "Stage B confusion matrix")
        plot_error_distance(stage_b_rows, out_dir / "stage_b_error_distance.png", "Stage B error distance")
        bad_stage_b = export_badcases_csv(stage_b_rows, out_dir / "stage_b_badcases.csv", top_n=args.top_n_badcases)
        print(f"[Stage B] rows={len(stage_b_rows)}, badcases={len(bad_stage_b)}")

    bad_graph: list[dict[str, Any]] = []
    if args.graph_predictions:
        graph_rows = read_jsonl(args.graph_predictions)
        plot_confusion_matrix(graph_rows, out_dir / "graph_confusion_matrix.png", "Graph verifier confusion matrix")
        plot_error_distance(graph_rows, out_dir / "graph_error_distance.png", "Graph verifier error distance")
        plot_subclaim_vs_error(graph_rows, out_dir / "graph_subclaim_vs_error.png")
        bad_graph = export_badcases_csv(graph_rows, out_dir / "graph_badcases.csv", top_n=args.top_n_badcases)
        print(f"[Graph verifier] rows={len(graph_rows)}, badcases={len(bad_graph)}")

    if args.graph_inputs:
        graph_inputs = read_jsonl(args.graph_inputs)
        summarize_stage_c_gating(graph_inputs, out_dir / "stage_c_gating_summary.png")
        if bad_graph:
            saved = visualize_badcase_graphs(bad_graph, graph_inputs, out_dir, max_cases=args.max_graph_cases)
            print(f"[Stage C graph] saved topology plots: {saved}")

    print(f"Visualization artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
