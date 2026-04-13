## 环境配置

```
conda create -p ../../conda/vllm092 python==3.11
conda activate /data/liaozijie/conda/vllm092
python -m pip install -U pip setuptools wheel
pip install -U --force-reinstall "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128" --index-url https://download.pytorch.org/whl/cu128
pip install -U --force-reinstall "vllm==0.9.2" "transformers==4.52.4" "tokenizers==0.21.1" "sentence-transformers==3.2.1"
pip install -r stage_cde/requirements.txt
```

## stage C

### 运行命令

`VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/build_graph_inputs.py --config configs/build_graph_inputs.yaml`