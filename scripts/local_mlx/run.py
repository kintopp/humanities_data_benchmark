#!/usr/bin/env python3
"""
Local MLX Benchmark Runner

Runs local MLX vision-language models against RISE humanities benchmarks
and outputs results in the existing framework's format.

Usage:
    python scripts/local_mlx/run.py                        # Interactive menu
    python scripts/local_mlx/run.py --model churro          # Single model
    python scripts/local_mlx/run.py --model churro,chandra  # Multiple models
    python scripts/local_mlx/run.py --model all             # All models
    python scripts/local_mlx/run.py --model churro --benchmark medieval_manuscripts
"""

import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image
import Levenshtein
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # humanities_data_benchmark/
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results"
VERSION_FILE = PROJECT_ROOT / "VERSION"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from local_mlx.models import MODEL_REGISTRY, BENCHMARK_SUFFIXES, get_test_id, get_model_path
from local_mlx.converters import convert


# ---------------------------------------------------------------------------
# Memory monitoring (macOS)
# ---------------------------------------------------------------------------

def get_memory_info() -> dict | None:
    """Get memory usage info on macOS. Returns dict with 'used_gb', 'free_gb', 'total_gb', 'pressure'."""
    try:
        import resource
        # Process RSS
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        process_mb = rusage.ru_maxrss / (1024 * 1024)  # macOS reports in bytes

        # System memory via sysctl
        result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
        total_bytes = int(result.stdout.strip().split()[-1])
        total_gb = total_bytes / (1024 ** 3)

        # Memory pressure
        result = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5)
        pressure = "unknown"
        for line in result.stdout.splitlines():
            if "System-wide memory free percentage" in line:
                pct = line.strip().split(":")[-1].strip().rstrip("%")
                try:
                    free_pct = int(pct)
                    pressure = "normal" if free_pct > 20 else "warning" if free_pct > 5 else "critical"
                    return {
                        "process_mb": process_mb,
                        "total_gb": total_gb,
                        "free_pct": free_pct,
                        "pressure": pressure,
                    }
                except ValueError:
                    pass
        return {"process_mb": process_mb, "total_gb": total_gb, "free_pct": -1, "pressure": pressure}
    except Exception:
        return None


def log_memory(label: str = ""):
    """Log current memory state."""
    info = get_memory_info()
    if info:
        prefix = f"  [{label}] " if label else "  "
        pct = f"{info['free_pct']}% free" if info['free_pct'] >= 0 else info['pressure']
        log.info(f"{prefix}Memory: process={info['process_mb']:.0f}MB, system={pct} ({info['pressure']})")
        if info['pressure'] == 'critical':
            log.warning(f"{prefix}⚠ CRITICAL memory pressure! Consider aborting.")
        return info
    return None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------

def get_benchmark_version() -> str:
    """Read version from VERSION file."""
    try:
        return VERSION_FILE.read_text().strip()
    except FileNotFoundError:
        return "unknown"


def get_git_commit() -> str:
    """Get the short git commit hash of the benchmark repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Scoring helpers (standalone, no Benchmark class needed)
# ---------------------------------------------------------------------------

def calculate_fuzzy_score(test_value, gold_value) -> float:
    """Fuzzy string similarity score (0.0–1.0) using rapidfuzz."""
    if test_value == gold_value:
        return 1.0
    if test_value is None or gold_value is None:
        return 0.0
    test_str = str(test_value)
    gold_str = str(gold_value)
    if test_str == gold_str:
        return 1.0
    if not isinstance(test_value, (str, int, float)) or not isinstance(gold_value, (str, int, float)):
        return 0.0
    return fuzz.ratio(test_str, gold_str) / 100.0


def normalize_empty(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and not value.strip():
        return ""
    return value


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = Levenshtein distance / len(reference)."""
    reference = normalize_empty(reference)
    hypothesis = normalize_empty(hypothesis)
    if not reference and not hypothesis:
        return 0.0
    if not reference or not hypothesis:
        return 1.0
    ref = " ".join(reference.lower().split())
    hyp = " ".join(hypothesis.lower().split())
    if ref == hyp:
        return 0.0
    return min(1.0, Levenshtein.distance(ref, hyp) / max(1, len(ref)))


# ---------------------------------------------------------------------------
# Scoring: medieval_manuscripts
# ---------------------------------------------------------------------------

def score_medieval_image(parsed: dict, ground_truth: dict) -> dict:
    """Score one image for medieval_manuscripts. Returns {"fuzzy": ..., "cer": ...}."""
    response_folios = parsed.get("folios", [])
    if not isinstance(response_folios, list):
        response_folios = []

    gt_sorted = sorted(ground_truth.items())
    results = []

    for idx, (folio_ref, gt_entries) in enumerate(gt_sorted):
        if not gt_entries:
            continue
        gt = gt_entries[0]
        resp = response_folios[idx] if idx < len(response_folios) and isinstance(response_folios[idx], dict) else None
        found = resp is not None

        # Folio number
        gt_folio = normalize_empty(gt.get("folio", ""))
        resp_folio = normalize_empty(resp.get("folio", "") if resp else "")
        if gt_folio or resp_folio:
            results.append({
                "similarity": calculate_fuzzy_score(resp_folio, gt_folio) if found else 0.0,
                "cer": calculate_cer(gt_folio, resp_folio) if found else 1.0,
            })

        # Main text
        gt_text = normalize_empty(gt.get("text", ""))
        resp_text = normalize_empty(resp.get("text", "") if resp else "")
        results.append({
            "similarity": calculate_fuzzy_score(resp_text, gt_text) if found else 0.0,
            "cer": calculate_cer(gt_text, resp_text) if found else 1.0,
        })

        # Additions
        for i in range(1, 10):
            key = f"addition{i}"
            gt_add = gt.get(key)
            if gt_add is None:
                break
            gt_add = normalize_empty(gt_add)
            resp_add = normalize_empty(resp.get(key, "") if resp else "")
            if not gt_add and not resp_add:
                continue
            results.append({
                "similarity": calculate_fuzzy_score(resp_add, gt_add) if found else 0.0,
                "cer": calculate_cer(gt_add, resp_add) if found else 1.0,
            })

    if not results:
        return {"fuzzy": 0.0, "cer": 1.0}

    avg_f = sum(r["similarity"] for r in results) / len(results)
    avg_c = sum(r["cer"] for r in results) / len(results)
    return {"fuzzy": round(avg_f, 3), "cer": round(avg_c, 3)}


# ---------------------------------------------------------------------------
# Scoring: fraktur_adverts
# ---------------------------------------------------------------------------

DEFAULT_SECTION = "Es wird zum Verkauf angetragen"
SECTION_MATCH_THRESHOLD = 0.90  # Relaxed from 0.95 to accommodate OCR spelling variants (e.g. ſ→s)


def _extract_number_prefix(text: str):
    m = re.match(r"^\s*(\d+)\.", text)
    return int(m.group(1)) if m else None


def _group_ads(ad_list: list, image_name: str = None) -> dict:
    grouped = defaultdict(dict)
    if not ad_list or not isinstance(ad_list, list):
        return grouped
    for ad in ad_list:
        if not isinstance(ad, dict):
            continue
        section = (ad.get("tags_section") or "").strip()
        if image_name == "image_4" and not section:
            section = DEFAULT_SECTION
            ad["tags_section"] = DEFAULT_SECTION
        number = _extract_number_prefix(ad.get("text", ""))
        if section and number:
            grouped[section][number] = ad
    return grouped


def score_fraktur_image(parsed: dict, ground_truth, image_name: str) -> dict:
    """Score one image for fraktur_adverts. Returns {"fuzzy": ..., "cer": ...}."""
    # Flatten ground truth
    if isinstance(ground_truth, dict):
        gt_flat = [e for ads in ground_truth.values() for e in ads]
    else:
        gt_flat = ground_truth

    # Build response grouped
    resp_ads = parsed.get("advertisements", [])
    if image_name == "image_4":
        for ad in resp_ads:
            if isinstance(ad, dict) and not ad.get("tags_section"):
                ad["tags_section"] = DEFAULT_SECTION
    resp_grouped = _group_ads(resp_ads, image_name)
    gt_grouped = _group_ads(gt_flat, image_name)

    results = []

    if image_name == "image_4" and DEFAULT_SECTION in gt_grouped:
        gt_section = gt_grouped[DEFAULT_SECTION]
        resp_section = resp_grouped.get(DEFAULT_SECTION, {})

        for number, gt_ad in gt_section.items():
            resp_ad = resp_section.get(number)
            if not resp_ad:
                for _, other_ads in resp_grouped.items():
                    if number in other_ads:
                        resp_ad = other_ads[number]
                        break
            if resp_ad:
                sim = calculate_fuzzy_score(resp_ad["text"], gt_ad["text"])
                cer_val = calculate_cer(gt_ad["text"], resp_ad["text"])
                results.append({"similarity": sim, "cer": cer_val})
            else:
                results.append({"similarity": 0.0, "cer": 1.0})
    else:
        for section, gt_ads in gt_grouped.items():
            resp_ads_section = resp_grouped.get(section, {})
            if not resp_ads_section:
                for rs, ra in resp_grouped.items():
                    if calculate_fuzzy_score(rs, section) >= SECTION_MATCH_THRESHOLD:
                        resp_ads_section = ra
                        break
            for number, gt_ad in gt_ads.items():
                resp_ad = resp_ads_section.get(number)
                if resp_ad:
                    sim = calculate_fuzzy_score(resp_ad["text"], gt_ad["text"])
                    cer_val = calculate_cer(gt_ad["text"], resp_ad["text"])
                    results.append({"similarity": sim, "cer": cer_val})
                else:
                    results.append({"similarity": 0.0, "cer": 1.0})

    if not results:
        return {"fuzzy": 0.0, "cer": 1.0}

    avg_f = sum(r["similarity"] for r in results) / len(results)
    avg_c = sum(r["cer"] for r in results) / len(results)
    return {"fuzzy": round(avg_f, 2), "cer": round(avg_c, 3)}


SCORE_FUNCTIONS = {
    "medieval_manuscripts": score_medieval_image,
    "fraktur_adverts": score_fraktur_image,
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_image_to_fit(image_path: str, max_size: int) -> str:
    """Resize an image if its longest side exceeds max_size. Returns path (may be temp)."""
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) <= max_size:
        return image_path

    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Save to a temp file (outside the images dir to avoid pickup)
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="mlx_resize_")
    os.close(fd)
    img.save(tmp_path)
    log.info(f"  Resized {w}x{h} → {new_w}x{new_h}")
    return tmp_path


def get_image_files(benchmark_name: str) -> list[tuple[str, str]]:
    """Return list of (basename, full_path) for all images in a benchmark."""
    images_dir = BENCHMARKS_DIR / benchmark_name / "images"
    if not images_dir.exists():
        return []
    files = []
    for f in sorted(images_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
            files.append((f.stem, str(f)))
    return files


def load_ground_truth(benchmark_name: str, basename: str) -> dict:
    """Load ground truth JSON for an image."""
    gt_path = BENCHMARKS_DIR / benchmark_name / "ground_truths" / f"{basename}.json"
    with open(gt_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# MLX inference
# ---------------------------------------------------------------------------

def build_conversation(model_cfg: dict) -> list[dict]:
    """
    Build a chat conversation structure for apply_chat_template.

    Follows the pattern from churro_cli.py: system prompt + user message
    with {"type": "image"} content for vision models.
    """
    system_prompt = model_cfg["system_prompt"]
    user_prompt = model_cfg["user_prompt"]

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build user content — always include image placeholder
    user_content = []
    user_content.append({"type": "image"})
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})

    messages.append({"role": "user", "content": user_content})
    return messages


def run_inference(model, processor, image_path: str, model_cfg: dict) -> tuple[str, float]:
    """
    Run a single inference. Returns (raw_text, duration_seconds).

    Uses processor.apply_chat_template (like churro_cli.py) for proper
    model-specific prompt formatting, then passes PIL Image in a list
    to generate().
    """
    from mlx_vlm import generate

    # Build conversation and format with chat template
    conversation = build_conversation(model_cfg)
    formatted_prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )

    # Load image as PIL (mlx_vlm expects PIL Image objects in a list)
    img = Image.open(image_path)

    kwargs = {
        "max_tokens": model_cfg["max_tokens"],
        "verbose": False,
    }

    # Temperature (mlx_vlm uses "temp" kwarg internally)
    if model_cfg["temperature"] is not None:
        kwargs["temp"] = model_cfg["temperature"]

    start = time.time()
    result = generate(
        model,
        processor,
        formatted_prompt,
        [img],
        **kwargs,
    )
    duration = time.time() - start

    # Handle both GenerationResult objects and plain strings
    text = result.text if hasattr(result, "text") else str(result)
    return text, duration


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_request_result(
    results_dir: Path,
    test_id: str,
    basename: str,
    raw_text: str,
    parsed: dict,
    score: dict,
    model_cfg: dict,
    duration: float,
    benchmark_version: str,
    benchmark_commit: str,
):
    """Save a per-image result file matching the existing framework format."""
    out_dir = results_dir / test_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "text": raw_text,
        "model": model_cfg["name"],
        "model_id": model_cfg["hf_id"],
        "provider": "local-mlx",
        "finish_reason": "stop",
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "estimated_cost_usd": 0.0,
        },
        "duration": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
        "parsed": parsed,
        "score": score,
        "benchmark_version": benchmark_version,
        "benchmark_commit": benchmark_commit,
    }

    filename = out_dir / f"request_{test_id}_{basename}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return filename


def save_scoring(
    results_dir: Path,
    test_id: str,
    all_scores: list[dict],
    model_cfg: dict,
    benchmark_version: str,
    benchmark_commit: str,
):
    """Save aggregate scoring.json matching the existing framework format."""
    out_dir = results_dir / test_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Average scores
    if all_scores:
        avg_fuzzy = sum(s["fuzzy"] for s in all_scores) / len(all_scores)
        avg_cer = sum(s["cer"] for s in all_scores) / len(all_scores)
    else:
        avg_fuzzy, avg_cer = 0.0, 1.0

    scoring = {
        "fuzzy": round(avg_fuzzy, 3),
        "cer": round(avg_cer, 3),
        "cost_summary": {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        },
        "model_id": model_cfg["hf_id"],
        "benchmark_version": benchmark_version,
        "benchmark_commit": benchmark_commit,
    }

    filename = out_dir / "scoring.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(scoring, f, indent=4, ensure_ascii=False)
    return filename


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def show_menu(benchmark_version: str, benchmark_commit: str) -> list[str]:
    """Display interactive model selection menu and return selected model keys."""
    models = list(MODEL_REGISTRY.items())

    # Build content lines first, then measure width for the box
    lines = []
    lines.append("")
    lines.append(f"  Benchmark suite: RISE Humanities Data Benchmark v{benchmark_version} ({benchmark_commit})")
    lines.append("")

    for i, (key, cfg) in enumerate(models, 1):
        lines.append(f"  {i}  {cfg['name']:<20s} {cfg['hf_id']:<45s} {cfg['params']:<6s} {cfg['quant']}")
        for bm in cfg["benchmarks"]:
            imgs = get_image_files(bm)
            lines.append(f"     \u2192 {bm} ({len(imgs)} imgs)")
        lines.append("")

    inner_w = max(len(line) for line in lines) + 2  # +2 for padding

    print()
    print(f"\u256d\u2500 Local MLX Benchmark Runner {'\u2500' * (inner_w - 28)}\u256e")
    for line in lines:
        print(f"\u2502{line:<{inner_w}}\u2502")
    print(f"\u2570{'\u2500' * inner_w}\u256f")
    print()

    selection = input("Select models to run (e.g. 1,3,4 or 'all', q to quit): ").strip()
    if selection.lower() in ("q", "quit", "exit"):
        return []
    if selection.lower() == "all":
        return list(MODEL_REGISTRY.keys())

    selected = []
    for part in selection.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(models):
                selected.append(models[idx][0])
            else:
                print(f"  Warning: {part} is out of range, skipping")
        except ValueError:
            if part.lower() in MODEL_REGISTRY:
                selected.append(part.lower())
            else:
                print(f"  Warning: '{part}' not recognized, skipping")

    return selected


def confirm_run(selected_models: list[str], benchmark_filter: str | None) -> bool:
    """Show confirmation summary before starting."""
    total_images = 0
    print()
    print("Run summary:")
    print("=" * 60)
    for key in selected_models:
        cfg = MODEL_REGISTRY[key]
        benchmarks = cfg["benchmarks"]
        if benchmark_filter:
            benchmarks = [b for b in benchmarks if b == benchmark_filter]
        for bm in benchmarks:
            imgs = get_image_files(bm)
            total_images += len(imgs)
            tid = get_test_id(key, bm)
            print(f"  {cfg['name']:<20s} {bm:<25s} {len(imgs):>3d} images  [{tid}]")
    print(f"{'':->60s}")
    print(f"  Total: {total_images} inference runs")
    print()
    resp = input("Proceed? [Y/n]: ").strip()
    return resp.lower() in ("", "y", "yes")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_model_on_benchmark(
    model_key: str,
    benchmark_name: str,
    date_str: str,
    benchmark_version: str,
    benchmark_commit: str,
):
    """Run a single model on a single benchmark. Returns aggregate score dict."""
    from mlx_vlm import load

    cfg = MODEL_REGISTRY[model_key]
    test_id = get_test_id(model_key, benchmark_name)
    model_path = get_model_path(model_key)
    results_dir = RESULTS_DIR / date_str

    images = get_image_files(benchmark_name)
    if not images:
        log.warning(f"No images found for {benchmark_name}")
        return None

    log.info(f"Loading model {cfg['name']} from {model_path}...")
    load_start = time.time()
    try:
        model, processor = load(model_path)
    except Exception as e:
        log.error(f"Failed to load model {cfg['name']}: {e}")
        return None
    load_time = time.time() - load_start
    log.info(f"  Model loaded in {load_time:.1f}s")
    log_memory("after load")

    all_scores = []

    for basename, image_path in images:
        log.info(f"  [{test_id}] Processing {basename}...")

        # Log image dimensions and memory before processing
        try:
            with Image.open(image_path) as _img:
                iw, ih = _img.size
                log.info(f"    Image size: {iw}x{ih} ({iw*ih/1e6:.1f} Mpx)")
        except Exception:
            pass
        mem = log_memory(basename)
        if mem and mem.get("pressure") == "critical":
            log.error(f"  Skipping {basename} — critical memory pressure!")
            all_scores.append({"fuzzy": 0.0, "cer": 1.0})
            continue

        # Resize if needed
        actual_path = image_path
        if cfg["max_image_size"]:
            actual_path = resize_image_to_fit(image_path, cfg["max_image_size"])

        # Run inference
        try:
            raw_text, duration = run_inference(model, processor, actual_path, cfg)
        except Exception as e:
            log.error(f"  Inference failed for {basename}: {e}")
            raw_text = ""
            duration = 0.0

        # Check memory after inference
        log_memory(f"{basename} post")

        # Clean up resized temp file
        if actual_path != image_path and os.path.exists(actual_path):
            os.remove(actual_path)

        # Post-process to benchmark JSON
        try:
            parsed = convert(cfg["output_format"], benchmark_name, raw_text)
        except Exception as e:
            log.error(f"  Conversion failed for {basename}: {e}")
            parsed = {"folios": [{"folio": "", "text": raw_text, "addition1": ""}]} \
                if benchmark_name == "medieval_manuscripts" \
                else {"advertisements": [{"date": None, "tags_section": None, "text": raw_text}]}

        # Load ground truth
        gt = load_ground_truth(benchmark_name, basename)

        # Score
        score_fn = SCORE_FUNCTIONS[benchmark_name]
        if benchmark_name == "fraktur_adverts":
            score = score_fn(parsed, gt, basename)
        else:
            score = score_fn(parsed, gt)

        log.info(f"    fuzzy={score['fuzzy']:.3f}  cer={score['cer']:.3f}  ({duration:.1f}s)")

        # Save per-image result
        save_request_result(
            results_dir, test_id, basename, raw_text, parsed, score,
            cfg, duration, benchmark_version, benchmark_commit,
        )
        all_scores.append(score)
        gc.collect()  # Free memory between images

    # Save aggregate scoring
    scoring_file = save_scoring(
        results_dir, test_id, all_scores, cfg,
        benchmark_version, benchmark_commit,
    )
    log.info(f"  Scoring saved to {scoring_file}")

    # Compute and return aggregate
    if all_scores:
        avg_fuzzy = sum(s["fuzzy"] for s in all_scores) / len(all_scores)
        avg_cer = sum(s["cer"] for s in all_scores) / len(all_scores)
    else:
        avg_fuzzy, avg_cer = 0.0, 1.0

    return {
        "model": cfg["name"],
        "benchmark": benchmark_name,
        "test_id": test_id,
        "fuzzy": round(avg_fuzzy, 3),
        "cer": round(avg_cer, 3),
        "images": len(images),
    }


def run_selected_models(selected_models: list[str], benchmark_filter: str | None = None):
    """Run all selected models on their compatible benchmarks."""
    benchmark_version = get_benchmark_version()
    benchmark_commit = get_git_commit()
    date_str = datetime.now().strftime("%Y-%m-%d")

    all_results = []

    for model_key in selected_models:
        cfg = MODEL_REGISTRY[model_key]
        benchmarks = cfg["benchmarks"]
        if benchmark_filter:
            benchmarks = [b for b in benchmarks if b == benchmark_filter]

        log.info(f"\n{'='*60}")
        log.info(f"Model: {cfg['name']} ({cfg['hf_id']})")
        log.info(f"{'='*60}")

        for bm in benchmarks:
            result = run_model_on_benchmark(
                model_key, bm, date_str, benchmark_version, benchmark_commit,
            )
            if result:
                all_results.append(result)

        # Free model memory before loading next
        log.info(f"Unloading {cfg['name']}...")
        gc.collect()

    # Print summary table
    print()
    print("=" * 80)
    print(f"{'Model':<22s} {'Benchmark':<25s} {'Fuzzy':>6s} {'CER':>6s} {'Images':>6s} {'Test ID'}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<22s} {r['benchmark']:<25s} {r['fuzzy']:>6.3f} {r['cer']:>6.3f} {r['images']:>6d} {r['test_id']}")
    print("=" * 80)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local MLX models on RISE humanities benchmarks",
    )
    parser.add_argument(
        "--model", "-m",
        help="Model(s) to run: comma-separated keys (churro,chandra,...) or 'all'",
    )
    parser.add_argument(
        "--benchmark", "-b",
        help="Run only this benchmark (medieval_manuscripts or fraktur_adverts)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_version = get_benchmark_version()
    benchmark_commit = get_git_commit()

    # Determine selected models
    if args.model:
        if args.model.lower() == "all":
            selected = list(MODEL_REGISTRY.keys())
        else:
            selected = []
            for part in args.model.split(","):
                key = part.strip().lower()
                if key in MODEL_REGISTRY:
                    selected.append(key)
                else:
                    print(f"Unknown model: {key}")
                    print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
                    sys.exit(1)
    else:
        # Interactive menu
        selected = show_menu(benchmark_version, benchmark_commit)

    if not selected:
        print("No models selected.")
        sys.exit(0)

    # Validate benchmark filter
    if args.benchmark:
        if args.benchmark not in BENCHMARK_SUFFIXES:
            print(f"Unknown benchmark: {args.benchmark}")
            print(f"Available: {', '.join(BENCHMARK_SUFFIXES.keys())}")
            sys.exit(1)

    # Confirm
    if not args.yes:
        if not confirm_run(selected, args.benchmark):
            print("Cancelled.")
            sys.exit(0)

    # Run
    run_selected_models(selected, args.benchmark)


if __name__ == "__main__":
    main()
