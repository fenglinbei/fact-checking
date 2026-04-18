import json

cases = []
with open("stage_cde/outputs/graph_inputs/test.graph.jsonl", "r") as f:
    for line in f:
        case = json.loads(line)
        cases.append(case)
        break

for case in cases[:1]:
    print(json.dumps(case, indent=2, ensure_ascii=False))