# Local MLX Benchmark Runner

Run local MLX vision-language models against RISE humanities benchmarks on Apple Silicon.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- Dependencies: `mlx-vlm`, `Pillow`, `rapidfuzz`, `python-Levenshtein`

```bash
pip install mlx-vlm Pillow rapidfuzz python-Levenshtein
```

## Models

| # | Model | HuggingFace ID | Params | Quant | Specialization |
|---|---|---|---|---|---|
| 1 | Churro | stanford-oval/churro-3B | 3B | 8-bit | Historical document OCR |
| 2 | Chandra | mlx-community/chandra-8bit | 9B | 8-bit | General VLM |
| 3 | olmOCR-2-7B | richardyoung/olmOCR-2-7B-1025-MLX-8bit | 7B | 8-bit | Document extraction |
| 4 | GLM-OCR | mlx-community/GLM-OCR-4bit | 0.9B | 4-bit | Document OCR |
| 5 | PaddleOCR-VL-1.5 | mlx-community/PaddleOCR-VL-1.5-bf16 | 0.9B | bf16 | Document/table OCR |
| 6 | LightOnOCR-2-1B | mlx-community/LightOnOCR-2-1B-bf16 | 1B | bf16 | Document OCR |

## Benchmarks

All 6 models run on:
- **medieval_manuscripts** (12 images) — 15th-century German handwritten transcription
- **fraktur_adverts** (5 images) — 18th-century Fraktur newspaper OCR

## Usage

### Interactive mode
```bash
python scripts/local_mlx/run.py
```

### Single model
```bash
python scripts/local_mlx/run.py --model churro
```

### Multiple models
```bash
python scripts/local_mlx/run.py --model churro,chandra,glmocr
```

### All models
```bash
python scripts/local_mlx/run.py --model all --yes
```

### Filter by benchmark
```bash
python scripts/local_mlx/run.py --model churro --benchmark medieval_manuscripts
```

## Output

Results are saved in the standard framework format under `results/{date}/{test_id}/`:

- `request_{test_id}_{image_name}.json` — Per-image results with raw text, parsed output, and scores
- `scoring.json` — Aggregate fuzzy and CER scores

### Test IDs

| Model | medieval_manuscripts | fraktur_adverts |
|---|---|---|
| Churro | TCHURRO_medieval | TCHURRO_fraktur |
| Chandra | TCHANDR_medieval | TCHANDR_fraktur |
| olmOCR | TOLMOCR_medieval | TOLMOCR_fraktur |
| GLM-OCR | TGLMOCR_medieval | TGLMOCR_fraktur |
| PaddleOCR | TPADDLE_medieval | TPADDLE_fraktur |
| LightOnOCR | TLIGHTN_medieval | TLIGHTN_fraktur |

## Architecture

```
scripts/local_mlx/
├── run.py          # Entry point: interactive menu, CLI, inference loop, scoring
├── models.py       # Model registry: HF IDs, prompts, temperature, benchmarks
├── converters.py   # Post-processors: XML/Markdown/text → benchmark JSON
└── README.md
```

Each model produces raw OCR output in its native format, then a deterministic post-processor converts it to the benchmark's expected JSON structure. This separates OCR quality from format compliance.
