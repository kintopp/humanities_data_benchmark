# Churro 3B (4-bit) vs GPT-5.2 — Fraktur Adverts OCR

Benchmark: `fraktur_adverts` from RISE Humanities Data Benchmark v0.4.0
Date: 2026-02-05
Hardware: Apple Silicon, 24 GB unified memory

## Results

| Image | Churro Fuzzy | GPT-5.2 Fuzzy | Churro CER | GPT-5.2 CER | Churro Time |
|-------|-------------|---------------|------------|-------------|-------------|
| image_1 (1209x2000) | **0.900** | 0.770 | **0.105** | 0.259 | 100.5s |
| image_2 (1209x2000) | **0.920** | 0.870 | **0.078** | 0.152 | 193.3s |
| image_3 (920x1149) | **0.960** | 0.960 | **0.031** | 0.047 | 70.5s |
| image_4 (925x1166) | 0.930 | **0.940** | **0.067** | 0.086 | 85.7s |
| image_5 (905x1163) | 0.910 | **0.930** | 0.124 | **0.110** | 80.1s |
| **Average** | **0.924** | 0.894 | **0.081** | 0.131 | 106.0s |

Total inference time: 530.1s (8 min 50s) for 5 images.

## Churro model parameters

| Parameter | Value |
|-----------|-------|
| Model | stanford-oval/churro-3B |
| Quantization | 4-bit (affine, group_size=64) |
| Weights on disk | 2.9 GB |
| Architecture | Qwen 2.5 VL 3B (36 layers, 16 attn heads, 2 KV heads) |
| System prompt | `Transcribe the entiretly of this historical documents to XML format.` |
| User prompt | (empty — image only) |
| Temperature | 0.6 |
| Max tokens | 20000 |
| Max image size | 2000px longest side |
| Output format | XML (post-processed to benchmark JSON) |
| Inference framework | mlx-vlm on Apple Silicon |

Note: the system prompt typos ("entiretly", "documents") are intentional — they match the fine-tuning data.

## Memory usage

| Stage | Process RSS | System free |
|-------|------------|-------------|
| After model load | 3,505 MB | 79% |
| During inference (peak) | 3,939 MB | 60% |
| Between images | 3,939 MB | 64% |

Peak system memory pressure stayed at "normal" throughout the run. No swap or memory warnings.

Full resolution (5000x8267, 41 Mpx) was attempted first but crashed with Metal GPU out-of-memory (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). 2500px also crashed. 2000px is the highest stable resolution on 24 GB.

## GPT-5.2 reference

| Parameter | Value |
|-----------|-------|
| Model | gpt-5.2-2025-12-11 |
| Test ID | T0492 |
| API cost | $0.134 total (5 images) |
| Run date | 2026-01-26 |

## Scoring methodology

Advertisements are matched between model output and ground truth by section heading (fuzzy-matched at 90% threshold) and numbered prefix (1., 2., etc.).

- **Fuzzy** (higher is better, 0–1): `rapidfuzz.fuzz.ratio` — normalized Levenshtein similarity.
- **CER** (lower is better, 0–1): Levenshtein edit distance / ground truth length, after lowercasing and whitespace normalization.
