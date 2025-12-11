#!/usr/bin/env python3
#
# Texture matching pipeline
# - Prompts for base directory (cross platform)
# - Checks dependencies before doing work
# - Finds filename matches between COM and Glide, copies Glide matches to glideout
# - Hashes GlideOut and NG files in parallel (phash) and finds best match within max_hamming
# - Copies COM files into NG folder structure for matched files
# - Produces:
#     * com_glide_matches.csv       (filename matches)
#     * glide_ng_hash_matches.csv   (best hash matches with hamming distances)
#     * full debug JSON              (all internal structures, errors, timings)
# - Adds tqdm progress bars and logs to a rotating log file
#
# Notes:
# - Uses ProcessPoolExecutor for hashing to utilize multiple CPUs.
# - Converts image hashes to hex strings for safe inter-process transfer.
# - Keeps backward compatible behavior and folder layout expectations.
#
# Usage:
#  python3 texture_pipeline.py
#

import os
import sys
import csv
import json
import shutil
import time
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ---------------------------
# Dependency check
# ---------------------------
missing = []
try:
    from PIL import Image
except Exception:
    missing.append("Pillow")
try:
    import imagehash
except Exception:
    missing.append("imagehash")
try:
    import numpy
except Exception:
    missing.append("numpy")
try:
    from tqdm import tqdm
except Exception:
    missing.append("tqdm")

if missing:
    print("Missing Python dependencies:")
    for m in missing:
        print(" - " + m)
    print("")
    print("Install with:")
    print("    pip install pillow imagehash numpy tqdm")
    sys.exit(1)

from PIL import Image
import imagehash
from tqdm import tqdm

# ---------------------------
# Logging setup
# ---------------------------

LOG_FILENAME = "texture_pipeline.log"
logger = logging.getLogger("texture_pipeline")
logger.setLevel(logging.DEBUG)

# Rotating file handler to avoid unbounded log growth
file_handler = RotatingFileHandler(LOG_FILENAME, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler for INFO+ messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info("Starting texture pipeline")

# ---------------------------
# Utility helpers
# ---------------------------

def write_csv(path, rows, headers):
    """Write list of dict rows to CSV file. Overwrites existing file."""
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        logger.info("Wrote CSV: %s", path)
    except Exception as e:
        logger.exception("Failed to write CSV %s: %s", path, e)

def write_json(path, obj):
    """Write JSON debug file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        logger.info("Wrote JSON debug: %s", path)
    except Exception as e:
        logger.exception("Failed to write JSON %s: %s", path, e)

def safe_listdir(path):
    """Return listdir or empty list on error."""
    try:
        return os.listdir(path)
    except Exception:
        return []

# ---------------------------
# Flatten directory (unchanged behavior)
# ---------------------------

def flatten_directory(source_dir, dest_dir):
    """
    Copy all PNG files from source_dir to dest_dir root, flattening structure.
    Resolve filename collisions by appending _N suffix.
    """
    logger.info("Flattening %s -> %s", source_dir, dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.lower().endswith(".png"):
                continue
            src = os.path.join(root, file)
            dst = os.path.join(dest_dir, file)
            base = dst
            i = 1
            while os.path.exists(dst):
                name, ext = os.path.splitext(base)
                dst = f"{name}_{i}{ext}"
                i += 1
            try:
                shutil.copy2(src, dst)
                count += 1
            except Exception as e:
                logger.exception("Failed to copy %s -> %s: %s", src, dst, e)
    logger.info("Flatten complete, files copied: %d", count)
    return count

# ---------------------------
# Filename matches COM <-> GLIDE
# ---------------------------

def find_filename_matches(com_dir, glide_dir, glideout_dir):
    """
    Find filenames present in both COM and Glide.
    Copy matched Glide files to glideout_dir (flattened).
    Return list of match records and a set of matched filenames.
    """
    logger.info("Finding filename matches between COM and Glide")
    os.makedirs(glideout_dir, exist_ok=True)

    com_files = set(f for f in safe_listdir(com_dir) if f.lower().endswith(".png"))
    logger.info("COM file count: %d", len(com_files))

    matches = []
    matched_filenames = set()
    # Walk glide_dir and copy matching filenames to glideout
    for root, _, files in os.walk(glide_dir):
        for file in files:
            if not file.lower().endswith(".png"):
                continue
            if file in com_files:
                src = os.path.join(root, file)
                dst = os.path.join(glideout_dir, file)
                base = dst
                i = 1
                while os.path.exists(dst):
                    name, ext = os.path.splitext(base)
                    dst = f"{name}_{i}{ext}"
                    i += 1
                try:
                    shutil.copy2(src, dst)
                    matches.append({
                        "filename": file,
                        "glide_source_path": src,
                        "copied_to": dst
                    })
                    matched_filenames.add(file)
                except Exception as e:
                    logger.exception("Failed to copy matched glide file %s -> %s: %s", src, dst, e)

    logger.info("Filename matches found: %d", len(matches))
    return matches, matched_filenames

# ---------------------------
# Parallel hashing helpers
# ---------------------------

def _hash_file_worker(path):
    """
    Worker executed in a separate process.
    Returns a tuple:
      (path, hex_hash, width, height, error_string_or_none)
    hex_hash is a lowercase hex string representation of the phash.
    """
    try:
        with Image.open(path) as img:
            # ensure RGB for consistency
            if img.mode != "RGB":
                img = img.convert("RGB")
            ph = imagehash.phash(img)
            # imagehash.ImageHash has __str__ yielding hex, use that
            hex_hash = ph.__str__()  # hex string
            w, h = img.size
            return (path, hex_hash, w, h, None)
    except Exception as e:
        return (path, None, None, None, str(e))

def hash_files_parallel(paths, workers=None, desc="Hashing files"):
    """
    Hash a list of file paths in parallel using ProcessPoolExecutor.
    Returns list of dicts: {path, hash, width, height, error}
    """
    results = []
    start = time.time()
    total = len(paths)
    logger.info("%s: %d files (workers=%s)", desc, total, workers or "auto")
    # Use number of CPUs by default
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_hash_file_worker, p): p for p in paths}
        for fut in tqdm(as_completed(futures), total=total, desc=desc, unit="file"):
            try:
                path, hex_hash, w, h, err = fut.result()
                results.append({
                    "path": path,
                    "hash": hex_hash,
                    "width": w,
                    "height": h,
                    "error": err
                })
            except Exception as e:
                # Unexpected exception from future.result()
                p = futures.get(fut, "<unknown>")
                logger.exception("Unexpected error hashing %s: %s", p, e)
                results.append({
                    "path": p,
                    "hash": None,
                    "width": None,
                    "height": None,
                    "error": str(e)
                })
    elapsed = time.time() - start
    logger.info("Hashing complete: %d items in %.2fs", len(results), elapsed)
    return results

# ---------------------------
# Build hash mapping between glideout and NG
# ---------------------------

def build_hash_mapping(glideout_dir, ng_dir, max_hamming=9, workers=None):
    """
    Hash NG and GlideOut files in parallel then find best NG match for each GlideOut
    within max_hamming distance.

    Returns:
      - mapping: dict glide_filename -> ng_relative_path
      - match_records: list of dict rows containing glide file, ng path, hamming, sizes, hashes
      - debug object capturing hashed lists and errors for JSON dump
    """
    logger.info("Building hash mapping (max_hamming=%d)", max_hamming)

    # Build file lists
    glideout_paths = [os.path.join(glideout_dir, f) for f in safe_listdir(glideout_dir) if f.lower().endswith(".png")]
    ng_paths = []
    for root, _, files in os.walk(ng_dir):
        for f in files:
            if f.lower().endswith(".png"):
                ng_paths.append(os.path.join(root, f))

    logger.info("GlideOut count: %d, NG count: %d", len(glideout_paths), len(ng_paths))

    # Hash both sets in parallel (separately)
    ng_hash_results = hash_files_parallel(ng_paths, workers=workers, desc="Hashing NG files")
    glide_hash_results = hash_files_parallel(glideout_paths, workers=workers, desc="Hashing GlideOut files")

    # Build lookup of NG hashes (hex -> list of entries) and store relative paths and sizes
    ng_lookup = {}
    ng_errors = []
    for r in ng_hash_results:
        if r["error"]:
            ng_errors.append({"path": r["path"], "error": r["error"]})
            continue
        hexh = r["hash"]
        rel = os.path.relpath(r["path"], ng_dir)
        ng_lookup.setdefault(hexh, []).append({
            "relpath": rel,
            "path": r["path"],
            "width": r["width"],
            "height": r["height"]
        })

    # Pre-cache list of NG hashes as imagehash objects for distance calc
    # Use tuple list [(hexstr, imagehash_obj, relpath, width, height)]
    ng_hash_items = []
    for hexh, entries in ng_lookup.items():
        try:
            ih = imagehash.hex_to_hash(hexh)
        except Exception:
            # fallback: try create phash from hex manually by ImageHash
            ih = None
        for ent in entries:
            ng_hash_items.append((hexh, ih, ent["relpath"], ent["width"], ent["height"]))

    # For each glide file, find best NG by Hamming (lowest) within max_hamming
    mapping = {}
    match_records = []
    glide_errors = []

    # For fast lookup of filename -> glide path
    glide_filename_to_path = {os.path.basename(r["path"]): r for r in glide_hash_results}

    for gr in tqdm(glide_hash_results, desc="Matching GlideOut -> NG", unit="file"):
        glide_path = gr["path"]
        glide_fname = os.path.basename(glide_path)
        if gr["error"]:
            glide_errors.append({"path": glide_path, "error": gr["error"]})
            continue
        glide_hex = gr["hash"]
        try:
            glide_ih = imagehash.hex_to_hash(glide_hex)
        except Exception:
            glide_ih = None

        best = None
        best_dist = None

        # iterate ng_hash_items and compute distance
        for ng_hex, ng_ih, ng_rel, ng_w, ng_h in ng_hash_items:
            if ng_ih is None or glide_ih is None:
                # If conversion to ImageHash failed, skip
                continue
            try:
                dist = int(glide_ih - ng_ih)
            except Exception:
                continue
            if dist <= max_hamming and (best_dist is None or dist < best_dist):
                best_dist = dist
                best = {
                    "ng_rel": ng_rel,
                    "ng_hex": ng_hex,
                    "ng_width": ng_w,
                    "ng_height": ng_h
                }
                # perfect match short-circuit
                if dist == 0:
                    break

        if best is not None:
            mapping[glide_fname] = best["ng_rel"]
            match_records.append({
                "glide_file": glide_fname,
                "glide_path": glide_path,
                "glide_hash": glide_hex,
                "glide_width": gr["width"],
                "glide_height": gr["height"],
                "ng_path": best["ng_rel"],
                "ng_hash": best["ng_hex"],
                "ng_width": best["ng_width"],
                "ng_height": best["ng_height"],
                "hamming_distance": best_dist
            })
            logger.debug("Matched %s -> %s (hamming %d)", glide_fname, best["ng_rel"], best_dist)
        else:
            # no acceptable match found within max_hamming
            logger.debug("No NG match within hamming %d for %s", max_hamming, glide_fname)

    debug = {
        "ng_hash_errors": ng_errors,
        "glide_hash_errors": glide_errors,
        "ng_hashed_count": len([r for r in ng_hash_results if not r["error"]]),
        "glide_hashed_count": len([r for r in glide_hash_results if not r["error"]])
    }

    logger.info("Hash matching complete. total matches: %d", len(match_records))
    return mapping, match_records, debug

# ---------------------------
# Convert COM files to NG structure
# ---------------------------

def convert_com_files(com_dir, mapping, comout_dir):
    """
    For each COM filename present in mapping, copy the COM file to comout_dir/
    with the NG relative path (creating directories as needed).
    Return counts and a list of conversion records for debug.
    """
    logger.info("Converting COM files to NG structure")
    os.makedirs(comout_dir, exist_ok=True)

    converted = 0
    skipped = 0
    conversions = []
    for f in safe_listdir(com_dir):
        if not f.lower().endswith(".png"):
            continue
        src = os.path.join(com_dir, f)
        if f not in mapping:
            skipped += 1
            continue
        ng_rel = mapping[f]
        dst = os.path.join(comout_dir, ng_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
            converted += 1
            conversions.append({"com_filename": f, "src": src, "dst": dst})
        except Exception as e:
            logger.exception("Failed to copy COM %s -> %s: %s", src, dst, e)

    logger.info("Converted: %d, Skipped: %d", converted, skipped)
    return {
        "converted": converted,
        "skipped": skipped,
        "conversions": conversions
    }

# ---------------------------
# Main orchestration
# ---------------------------

def main():
    logger.info("Prompting for base directory")
    print("Enter base directory (contains com, glide, ng). Example windows: C:\\Users\\You\\wip or linux: /home/user/wip")
    base = input("Base directory: ").strip()
    if not base:
        logger.error("No base directory provided, exiting")
        print("No base directory provided. Exiting.")
        return
    base = os.path.expanduser(base)
    if not os.path.isdir(base):
        logger.error("Base directory does not exist: %s", base)
        print("Directory does not exist:", base)
        return

    # Standard locations under base
    com_dir = os.path.join(base, "com")
    glide_dir = os.path.join(base, "glide")
    ng_dir = os.path.join(base, "ng")
    glideout_dir = os.path.join(base, "glideout")
    comout_dir = os.path.join(base, "comout")

    # Output file paths
    csv_com_glide = os.path.join(base, "com_glide_matches.csv")
    csv_ng_glide = os.path.join(base, "glide_ng_hash_matches.csv")
    debug_json = os.path.join(base, "texture_pipeline_debug.json")

    # Validate inputs
    for p in (com_dir, glide_dir, ng_dir):
        if not os.path.isdir(p):
            logger.error("Required directory missing: %s", p)
            print("Required directory not found:", p)
            return

    # Flatten options: currently we assume com and glide are already flat per original design.
    # If the user wants flattening, they can call flatten_directory separately or we can add a flag.
    # Step 1: filename matches
    t0 = time.time()
    filename_matches, matched_filenames = find_filename_matches(com_dir, glide_dir, glideout_dir)
    write_csv(csv_com_glide, filename_matches, ["filename", "glide_source_path", "copied_to"])
    t1 = time.time()
    logger.info("Filename matching time: %.2fs", t1 - t0)

    if len(filename_matches) == 0:
        logger.warning("No filename matches found. Nothing to do.")
        print("No filename matches found. Exiting.")
        return

    # Step 2: build hash mapping (parallel)
    t0 = time.time()
    # Auto choose worker count: let ProcessPoolExecutor decide by default (None)
    mapping, match_records, hash_debug = build_hash_mapping(glideout_dir, ng_dir, max_hamming=9, workers=None)
    t1 = time.time()
    logger.info("Hash mapping time: %.2fs", t1 - t0)

    # Write CSV of hash matches
    rows = []
    for rec in match_records:
        rows.append({
            "glide_file": rec["glide_file"],
            "glide_path": rec["glide_path"],
            "glide_hash": rec["glide_hash"],
            "glide_width": rec.get("glide_width"),
            "glide_height": rec.get("glide_height"),
            "ng_path": rec["ng_path"],
            "ng_hash": rec.get("ng_hash"),
            "ng_width": rec.get("ng_width"),
            "ng_height": rec.get("ng_height"),
            "hamming_distance": rec["hamming_distance"]
        })
    write_csv(csv_ng_glide, rows, ["glide_file", "glide_path", "glide_hash", "glide_width", "glide_height", "ng_path", "ng_hash", "ng_width", "ng_height", "hamming_distance"])

    # Step 3: convert COM files
    t0 = time.time()
    conversion_info = convert_com_files(com_dir, mapping, comout_dir)
    t1 = time.time()
    logger.info("COM conversion time: %.2fs", t1 - t0)

    # Final: write full debug JSON
    debug = {
        "base_dir": base,
        "counts": {
            "com_total": len([f for f in safe_listdir(com_dir) if f.lower().endswith(".png")]),
            "glide_total": len([f for f in safe_listdir(glide_dir) if f.lower().endswith(".png")]),
            "glideout_total": len([f for f in safe_listdir(glideout_dir) if f.lower().endswith(".png")]),
            "ng_total": sum(1 for root, _, files in os.walk(ng_dir) for f in files if f.lower().endswith(".png")),
            "filename_matches": len(filename_matches),
            "hash_matches": len(match_records),
            "converted": conversion_info["converted"],
            "skipped": conversion_info["skipped"]
        },
        "filename_matches": filename_matches,
        "hash_matches": match_records,
        "conversion": conversion_info,
        "hash_debug": hash_debug,
        "log_file": os.path.abspath(LOG_FILENAME),
        "csv_files": {
            "com_glide_csv": os.path.abspath(csv_com_glide),
            "glide_ng_csv": os.path.abspath(csv_ng_glide)
        }
    }
    write_json(debug_json, debug)

    logger.info("Pipeline complete. Outputs:")
    logger.info("  COM->Glide CSV: %s", csv_com_glide)
    logger.info("  Glide->NG CSV: %s", csv_ng_glide)
    logger.info("  COM converted output dir: %s", comout_dir)
    logger.info("  Debug JSON: %s", debug_json)
    print("Done. See log and debug JSON for details.")

if __name__ == "__main__":
    main()
