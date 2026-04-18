import json

def main():
    cases = []
    with open('stage_ab/outputs/liar-raw/stage_a/stage_a_test.jsonl', 'r') as f:
        for line in f:
            cases.append(json.loads(line))

    for item in cases[:1]:
        print(json.dumps(item, indent=2, ensure_ascii=False))
        print('---')

if __name__ == '__main__':
    main()