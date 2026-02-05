"""
Model registry for local MLX vision-language models.

Each entry defines a model's identity, inference configuration,
output format, and compatible benchmarks.
"""

# olmOCR prompt (from official documentation)
OLMOCR_PROMPT = (
    "Below is the image of one page of a document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "Do NOT include any extra commentary, headers, or formatting instructions.\n"
    "If there are any images, charts, or other non-text elements, "
    "describe them briefly in square brackets.\n"
    "Preserve the original structure as much as possible, "
    "including line breaks and paragraph separations where appropriate."
)

MODEL_REGISTRY = {
    "churro": {
        "name": "Churro",
        "hf_id": "stanford-oval/churro-3B",
        "local_path": "/Users/bosse0000/Documents/experiments/churro-mlx/mlx_churro_8bit",
        "params": "3B",
        "quant": "8-bit",
        "system_prompt": "Transcribe the entiretly of this historical documents to XML format.",
        "user_prompt": "",  # Image only — no text prompt
        "temperature": 0.6,
        "max_tokens": 20000,
        "output_format": "xml",
        "max_image_size": 1500,
        "benchmarks": ["medieval_manuscripts", "fraktur_adverts"],
        "test_id_prefix": "TCHURRO",
    },
    "churro4": {
        "name": "Churro-4bit",
        "hf_id": "stanford-oval/churro-3B",
        "local_path": "/Users/bosse0000/Documents/experiments/churro-mlx/mlx_churro_4bit",
        "params": "3B",
        "quant": "4-bit",
        "system_prompt": "Transcribe the entiretly of this historical documents to XML format.",
        "user_prompt": "",  # Image only — no text prompt
        "temperature": 0.6,
        "max_tokens": 20000,
        "output_format": "xml",
        "max_image_size": 2000,  # Higher than 8-bit (1500); 2500 still OOMs on Metal GPU
        "benchmarks": ["medieval_manuscripts", "fraktur_adverts"],
        "test_id_prefix": "TCHUR4B",
    },
    "chandra": {
        "name": "Chandra",
        "hf_id": "mlx-community/chandra-8bit",
        "local_path": None,
        "params": "9B",
        "quant": "8-bit",
        "system_prompt": None,
        "user_prompt": "convert to markdown",
        "temperature": None,  # Use model default
        "max_tokens": 2048,
        "output_format": "markdown",
        "max_image_size": None,
        "benchmarks": ["medieval_manuscripts", "fraktur_adverts"],
        "test_id_prefix": "TCHANDR",
    },
    "olmocr": {
        "name": "olmOCR-2-7B",
        "hf_id": "richardyoung/olmOCR-2-7B-1025-MLX-8bit",
        "local_path": None,
        "params": "7B",
        "quant": "8-bit",
        "system_prompt": None,
        "user_prompt": OLMOCR_PROMPT,
        "temperature": None,
        "max_tokens": 2048,
        "output_format": "markdown",
        "max_image_size": None,
        "benchmarks": ["medieval_manuscripts", "fraktur_adverts"],
        "test_id_prefix": "TOLMOCR",
    },
    "glmocr": {
        "name": "GLM-OCR",
        "hf_id": "mlx-community/GLM-OCR-4bit",
        "local_path": None,
        "params": "0.9B",
        "quant": "4-bit",
        "system_prompt": None,
        "user_prompt": "Extract all text from this image and format as plain text.",
        "temperature": 0.2,
        "max_tokens": 4096,
        "output_format": "text",
        "max_image_size": None,
        "benchmarks": ["medieval_manuscripts", "fraktur_adverts"],
        "test_id_prefix": "TGLMOCR",
    },
    "paddleocr": {
        "name": "PaddleOCR-VL-1.5",
        "hf_id": "mlx-community/PaddleOCR-VL-1.5-bf16",
        "local_path": None,
        "params": "0.9B",
        "quant": "bf16",
        "system_prompt": None,
        "user_prompt": "Extract all text from this image and format as plain text.",
        "temperature": 0.2,
        "max_tokens": 4096,
        "output_format": "text",
        "max_image_size": None,
        "benchmarks": ["medieval_manuscripts", "fraktur_adverts"],
        "test_id_prefix": "TPADDLE",
    },
    "lightonocr": {
        "name": "LightOnOCR-2-1B",
        "hf_id": "mlx-community/LightOnOCR-2-1B-bf16",
        "local_path": None,
        "params": "1B",
        "quant": "bf16",
        "system_prompt": None,
        "user_prompt": "",  # Empty prompt required
        "temperature": 0.2,
        "max_tokens": 4096,
        "output_format": "text",
        "max_image_size": None,
        "benchmarks": ["medieval_manuscripts", "fraktur_adverts"],
        "test_id_prefix": "TLIGHTN",
    },
}

# Map benchmark names to their test ID suffixes
BENCHMARK_SUFFIXES = {
    "medieval_manuscripts": "medieval",
    "fraktur_adverts": "fraktur",
}


def get_test_id(model_key: str, benchmark_name: str) -> str:
    """Generate a test ID like TCHURRO_medieval."""
    model = MODEL_REGISTRY[model_key]
    suffix = BENCHMARK_SUFFIXES[benchmark_name]
    return f"{model['test_id_prefix']}_{suffix}"


def get_model_path(model_key: str) -> str:
    """Return local path if available, otherwise HuggingFace ID for download."""
    model = MODEL_REGISTRY[model_key]
    if model["local_path"]:
        return model["local_path"]
    return model["hf_id"]
