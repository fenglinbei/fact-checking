## 环境配置

```
conda create -p ../../conda/vllm092 python==3.11
conda activate /data/liaozijie/conda/vllm092
python -m pip install -U pip setuptools wheel
python -m pip install "torch==2.7.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m pip install "vllm==0.9.2" --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r stage_cde/requirements.txt
```

## stage C

### 运行命令

`VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/build_graph_inputs.py --config configs/build_graph_inputs.yaml`