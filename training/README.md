# Training

This directory contains the maintainable version of the Colab fine-tuning
workflow. It trains a LoRA adapter with Unsloth and can optionally export or
push a GGUF model.

## Colab Setup

First confirm the runtime has a CUDA GPU:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

If this prints `False`, switch to `Runtime > Change runtime type > GPU`, restart
the runtime, and rerun setup.

Install the GPU training dependencies in a Colab cell before running the script:

```bash
pip install -U pip
pip install unsloth
pip install trl peft accelerate bitsandbytes
```

Do not pin old `xformers` versions in Colab. If pip starts `Building wheels for
collected packages: xformers`, stop the cell and use the commands above.

## Train From Hugging Face Dataset

```bash
python training/train_unsloth.py \
  --dataset seankwalker/seanbot-2-imessage \
  --max-steps 60 \
  --output-dir outputs
```

## Train From Local Export

Use the JSONL output from `imessage/main.py` when possible:

```bash
python training/train_unsloth.py \
  --dataset-file training_pairs.jsonl \
  --num-train-epochs 1 \
  --output-dir outputs
```

CSV files with `input` and `output` columns are also supported.

The default training settings are T4-oriented: `--max-seq-length 1024`,
`--per-device-train-batch-size 1`, and `--gradient-accumulation-steps 8`.
If an A100/L4 has headroom, increase `--max-seq-length` or per-device batch
size. If a T4 still runs out of memory, keep batch size at `1` and reduce
`--max-seq-length` to `768`.

## Sample After Training

```bash
python training/train_unsloth.py \
  --dataset-file training_pairs.jsonl \
  --max-steps 60 \
  --sample-prompt "what are you up to?"
```

## GGUF Export

For local export:

```bash
python training/train_unsloth.py \
  --dataset-file training_pairs.jsonl \
  --save-gguf-dir model \
  --gguf-quantization q8_0
```

For Hugging Face upload, export `HF_TOKEN` in the environment first:

```bash
python training/train_unsloth.py \
  --dataset-file training_pairs.jsonl \
  --push-gguf \
  --hub-model-id seankwalker/seanbot-2-llama-3-1-imessage
```

The script reads `HF_TOKEN` from the environment when `--push-gguf` is used. It
does not read `.env` files automatically.

## Privacy

Training datasets, checkpoints, generated samples, and pushed models may encode
private conversation content. Keep generated artifacts out of Git and review
uploads before sharing.
