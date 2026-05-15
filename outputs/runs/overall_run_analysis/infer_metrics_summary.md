# Overall Run Analysis

## Infer Metrics Summary

- All API metric records: `infer_metrics_summary_all.csv`
- Latest record per source/top_k/split/checkpoint: `infer_metrics_summary_latest.csv`
- Line chart: `infer_metrics_line_chart.png`
- Prompt statistics table: `prompt_stats_summary.csv`
- Prompt statistics chart: `prompt_stats_line_chart.png`
- Total records: 34
- Latest rows: 30

## Included Artifacts

- Overall infer metrics table: `infer_metrics_summary_latest.csv`
- Overall infer metrics chart: `infer_metrics_line_chart.png`
- Prompt statistics table: `prompt_stats_summary.csv`
- Prompt statistics chart: `prompt_stats_line_chart.png`
- b3 1024 top_k=0..8 test table: `b3_mmr_topk_test_curves_1024/test_metrics_top_k_0_8.csv`
- b3 1024 top_k=0..8 test chart: `b3_mmr_topk_test_curves_1024/test_metrics_top_k_0_8.png`

## Latest Records

| source_root | top_k | mmr_lambda | split | checkpoint | infer_id | num_samples | accuracy | macro_precision | macro_recall | macro_f1 | parse_error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b3_mmr_topk_sweep_1024 | 0 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.265387689848 | 0.238921534978 | 0.271380912525 | 0.207291624831 | 0 |
| b3_mmr_topk_sweep_1024 | 1 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.275779376499 | 0.276069430855 | 0.276876963743 | 0.266730448869 | 0 |
| b3_mmr_topk_sweep_1024 | 2 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.270183852918 | 0.272108124067 | 0.287341880373 | 0.273158577956 | 0 |
| b3_mmr_topk_sweep_1024 | 3 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.270983213429 | 0.282549538285 | 0.282635570845 | 0.277109970951 | 0 |
| b3_mmr_topk_sweep_1024 | 4 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.270983213429 | 0.287828214901 | 0.278135568725 | 0.272898449318 | 0 |
| b3_mmr_topk_sweep_1024 | 5 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.270183852918 | 0.281320068591 | 0.284368374867 | 0.276910921229 | 0 |
| b3_mmr_topk_sweep_1024 | 6 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.268585131894 | 0.282078870595 | 0.277226930435 | 0.274636776156 | 0 |
| b3_mmr_topk_sweep_1024 | 7 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.274980015987 | 0.291071519017 | 0.279364308594 | 0.274416498677 | 0 |
| b3_mmr_topk_sweep_1024 | 8 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.274980015987 | 0.278515433387 | 0.275364128205 | 0.267337326841 | 0 |
| heuristic_lambda_mmr | 5 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.27657873701 | 0.280514171339 | 0.292781034032 | 0.279905240209 | 0 |
| heuristic_lambda_mmr_fullft | 5 | 0.7 | test | best | 0d27dabf11a7 | 1251 | 0.310951239009 | 0.328642104839 | 0.33454807929 | 0.321463216342 | 0 |
| mmr_sensitivity_gated | 5 | 0.7 | test | best | 79d8b34809bb | 1251 | 0.274180655476 | 0.291765574852 | 0.287190134297 | 0.279483490943 | 0 |
| mmr_topk_sweep_infer | 0 | 0.7 | test | best | 9a002ceb9c81 | 1251 | 0.123101518785 | 0.166341467509 | 0.194351183815 | 0.088475236792 | 0 |
| mmr_topk_sweep_infer | 2 | 0.7 | test | best | ba745f87dcd7 | 1251 | 0.224620303757 | 0.233711942291 | 0.225926195262 | 0.21635384472 | 0 |
| mmr_topk_sweep_infer | 4 | 0.7 | test | best | f4015acb91b1 | 1251 | 0.238209432454 | 0.252569126852 | 0.235784839781 | 0.230595341428 | 0 |
| mmr_topk_sweep_infer | 6 | 0.7 | test | best | bd583eff3efe | 1251 | 0.242206235012 | 0.286079657515 | 0.242929170253 | 0.241983744364 | 0 |
| mmr_topk_sweep_infer | 8 | 0.7 | test | best | e7ce47ec59f2 | 1251 | 0.256594724221 | 0.291052195531 | 0.249581569795 | 0.250954803723 | 0 |
| mmr_topk_sweep_infer | 10 | 0.7 | test | best | f7aacf52ba20 | 1251 | 0.240607513989 | 0.263883095494 | 0.236902992335 | 0.236023868902 | 0 |
| mmr_topk_sweep_infer | 12 | 0.7 | test | best | f19ed0a01475 | 1251 | 0.254996003197 | 0.273334547012 | 0.250234689137 | 0.248786256569 | 0 |
| mmr_topk_sweep_infer | 14 | 0.7 | test | best | 558de893f23f | 1251 | 0.258992805755 | 0.282244426332 | 0.253688727019 | 0.254943862049 | 0 |
| mmr_topk_sweep_infer | 16 | 0.7 | test | best | 81def46d7ce8 | 1251 | 0.26618705036 | 0.290193121601 | 0.258470519024 | 0.262024107026 | 0 |
| mmr_topk_sweep_infer | 18 | 0.7 | test | best | 2bf8b25abbbe | 1251 | 0.274180655476 | 0.297802894137 | 0.262969473081 | 0.265705945228 | 0 |
| mmr_topk_sweep_infer | 20 | 0.7 | test | best | b2358170fa12 | 1251 | 0.27657873701 | 0.310481528795 | 0.26873451272 | 0.272947067349 | 0 |
| mmr_topk_sweep_infer | 22 | 0.7 | test | best | e270cb01c727 | 1251 | 0.262190247802 | 0.288391379648 | 0.252491577497 | 0.253595993302 | 0 |
| mmr_topk_sweep_infer | 24 | 0.7 | test | best | 8311d51dd899 | 1251 | 0.270983213429 | 0.29500292883 | 0.26189839944 | 0.261845007651 | 0 |
| mmr_topk_sweep_infer | 26 | 0.7 | test | best | 30b0ec85c3a3 | 1251 | 0.266986410871 | 0.302853766682 | 0.26034781404 | 0.262384284645 | 0 |
| mmr_topk_sweep_infer | 28 | 0.7 | test | best | 82107877f01b | 1251 | 0.266986410871 | 0.290887810066 | 0.256991489636 | 0.256897146129 | 0 |
| mmr_topk_sweep_infer | 30 | 0.7 | test | best | 9e58df364083 | 1251 | 0.262989608313 | 0.292991672701 | 0.254903299722 | 0.255661670915 | 0 |
| mmr_topk_sweep_infer | 32 | 0.7 | test | best | ac476cc8ac43 | 1251 | 0.264588329337 | 0.295400166791 | 0.256323217283 | 0.258026832279 | 0 |
| reranker_only | 5 |  | test | best | 79d8b34809bb | 1251 | 0.269384492406 | 0.290776006788 | 0.276544016805 | 0.274531132653 | 0 |

## Metric Leaders

| metric | best | runner_up | delta |
| --- | --- | --- | --- |
| accuracy | heuristic_lambda_mmr_fullft top_k=5: 0.310951239009 | heuristic_lambda_mmr top_k=5: 0.27657873701 | 0.034373 |
| macro_precision | heuristic_lambda_mmr_fullft top_k=5: 0.328642104839 | mmr_topk_sweep_infer top_k=20: 0.310481528795 | 0.018161 |
| macro_recall | heuristic_lambda_mmr_fullft top_k=5: 0.33454807929 | heuristic_lambda_mmr top_k=5: 0.292781034032 | 0.041767 |
| macro_f1 | heuristic_lambda_mmr_fullft top_k=5: 0.321463216342 | heuristic_lambda_mmr top_k=5: 0.279905240209 | 0.041558 |

## Sensitivity-Gated Details

| source_root | run_name | top_k | mmr_lambda | theta_s | theta_r | lambda_low | gating_mode | epsilon | accuracy | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmr_sensitivity_gated | ts0p8_tr0p3_ll0p2_basic__9a7f2ee7 | 5 | 0.7 | 0.8 | 0.3 | 0.2 | basic |  | 0.274180655476 | 0.279483490943 |

## Missing Prompt Stats

These runs are missing `train/prompt_stats/prompt_stats.json`; prompt-stat panels skip them.

| source_root | run_name | top_k | expected_path |
| --- | --- | --- | --- |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-0__8283349c | 0 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-0__8283349c/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-2__ab9ab486 | 2 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-2__ab9ab486/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-4__21b2059b | 4 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-4__21b2059b/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-6__93d6f740 | 6 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-6__93d6f740/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-8__9e605e05 | 8 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-8__9e605e05/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-10__09382603 | 10 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-10__09382603/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-12__21d39f02 | 12 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-12__21d39f02/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-14__ab3b17c8 | 14 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-14__ab3b17c8/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-16__bdc64fdd | 16 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-16__bdc64fdd/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-18__848f6a1a | 18 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-18__848f6a1a/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-20__191bb752 | 20 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-20__191bb752/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-22__ede2b82b | 22 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-22__ede2b82b/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-24__89425e76 | 24 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-24__89425e76/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-26__8b5ae44e | 26 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-26__8b5ae44e/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-28__c696ff2c | 28 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-28__c696ff2c/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-30__86ba27ab | 30 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-30__86ba27ab/train/prompt_stats/prompt_stats.json |
| mmr_topk_sweep_infer | build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-32__b4ea4992 | 32 | /home/fenglin/project/fact-checking/outputs/runs/mmr_topk_sweep_infer/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-32__b4ea4992/train/prompt_stats/prompt_stats.json |

## Duplicate Groups

| source_root | top_k | mmr_lambda | split | checkpoint | count | infer_ids |
| --- | --- | --- | --- | --- | --- | --- |
| mmr_topk_sweep_infer | 0 | 0.7 | test | best | 2 | 98a1917c4eb9, 9a002ceb9c81 |
| mmr_topk_sweep_infer | 2 | 0.7 | test | best | 2 | 7cefc941117a, ba745f87dcd7 |
| mmr_topk_sweep_infer | 4 | 0.7 | test | best | 2 | d2c71894aef4, f4015acb91b1 |
| mmr_topk_sweep_infer | 6 | 0.7 | test | best | 2 | 386a61d4fdb4, bd583eff3efe |
