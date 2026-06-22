PYTHON_BIN=/data/liaozijie/conda/accelerate-fc/bin/python
export PYTHONPATH=src
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY first}"

RUN_ROOT=outputs/selectors/atom_anchor/liar_raw_abc_v0_1
CHUNK_CACHE=outputs/cache/chunk_mmr/d4cbf7c18126
TRAIN_ORACLE=outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl

for SPLIT in train test; do
  RAW_PATH=data/raw/LIAR-RAW/${SPLIT}.json

  # 1. Claim atomization
  "$PYTHON_BIN" scripts/phase5_selectors/build/generate_claim_atom_cache.py \
    --input-mode raw_split \
    --raw-path "$RAW_PATH" \
    --dataset liar_raw \
    --label-schema liar6 \
    --split "$SPLIT" \
    --output-dir "$RUN_ROOT/01_claim_atoms" \
    --atom-cache-dir "$RUN_ROOT/01_claim_atoms/cache" \
    --atom-model deepseek-v4-flash \
    --atom-base-url https://api.deepseek.com \
    --atom-api-key-env DEEPSEEK_API_KEY \
    --api-concurrency 128 \
    --max-tokens 2048 \
    --top-p 1.0 \
    --thinking-type disabled \
    --resume-atoms

  # 2. Atom-conditioned retrieval
  ORACLE_ARG=()
  if [ "$SPLIT" = "train" ]; then
    ORACLE_ARG=(--oracle-results "$TRAIN_ORACLE")
  else
    ORACLE_ARG=(--oracle-results "")
  fi

  "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_conditioned_retrieval.py \
    --claim-atoms-jsonl "$RUN_ROOT/01_claim_atoms/claim_atoms_${SPLIT}.jsonl" \
    --chunk-cache-path "$CHUNK_CACHE/${SPLIT}.pkl" \
    --split "$SPLIT" \
    --output-dir "$RUN_ROOT/02_atom_retrieval" \
    --embedder-model /data/models/bge-base-en-v1.5 \
    --device cuda \
    --per-atom-keep 20 \
    --merged-pool-size 20 \
    --selector-top-k 5 \
    "${ORACLE_ARG[@]}"

  # 3. Atom union
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_retrieval_union.py \
    --baseline-jsonl "$RUN_ROOT/02_atom_retrieval/baseline_claim_mmr_selected_${SPLIT}.jsonl" \
    --atom-pool-jsonl "$RUN_ROOT/02_atom_retrieval/merged_candidate_pool_${SPLIT}.jsonl" \
    --split "$SPLIT" \
    --output-dir "$RUN_ROOT/03_atom_union" \
    --selector-top-k 5 \
    "${ORACLE_ARG[@]}"

  # 4. Evidence-map candidate pool
  PREP_ORACLE_ARG=()
  if [ "$SPLIT" = "train" ]; then
    PREP_ORACLE_ARG=(--oracle-results "$TRAIN_ORACLE")
  fi

  "$PYTHON_BIN" scripts/phase5_selectors/build/prepare_evidence_map_candidate_pool.py \
    --input-candidate-file "$RUN_ROOT/03_atom_union/atom_union_candidate_pool_${SPLIT}.jsonl" \
    --output-dir "$RUN_ROOT/04_evidence_map" \
    --split "$SPLIT" \
    --candidate-source atom_union \
    --candidate-top-n 20 \
    "${PREP_ORACLE_ARG[@]}"

  # 5. Evidence annotation
  "$PYTHON_BIN" scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
    --candidate-pool "$RUN_ROOT/04_evidence_map/evidence_map_candidate_pool_${SPLIT}.jsonl" \
    --output-dir "$RUN_ROOT/04_evidence_map" \
    --split "$SPLIT" \
    --prompt-version atom_evidence_map_v0_1 \
    --model deepseek-v4-flash \
    --api-key-env DEEPSEEK_API_KEY \
    --concurrency 128 \
    --requests-per-minute 2048 \
    --max-tokens 8192 \
    --top-p 1.0 \
    --thinking-type disabled \
    --resume

  # 6. Evidence-map postprocess
  "$PYTHON_BIN" scripts/phase5_selectors/build/postprocess_evidence_maps.py \
    --candidate-pool "$RUN_ROOT/04_evidence_map/evidence_map_candidate_pool_${SPLIT}.jsonl" \
    --annotations "$RUN_ROOT/04_evidence_map/deepseek_evidence_map_annotations_${SPLIT}.jsonl" \
    --output-dir "$RUN_ROOT/04_evidence_map" \
    --split "$SPLIT"

  # 7. MREC trace
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_mrec_traces.py \
    --input "$RUN_ROOT/04_evidence_map/candidate_evidence_map_features_${SPLIT}.jsonl" \
    --output-dir "$RUN_ROOT/05_mrec" \
    --split "$SPLIT" \
    --candidate-top-n 20 \
    --max-steps 10 \
    --target-resolved-rate 0.80 \
    --cue-policy atom_proposition \
    --selector-name mrec_greedy_transition_v0_1
done