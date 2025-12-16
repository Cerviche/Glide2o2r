#!/usr/bin/env python3
"""
Enhanced Texture Hash Generator for Next-Gen Pipeline
=====================================================

Step 1 of 3 in the enhanced Glide → SoH texture conversion pipeline.

This script scans a folder of PNG textures and computes hashes for use in
enhanced texture mapping. It supports multiple hashing algorithms, alpha
channel hashing, exact duplicate detection, and parallel processing.

Key Features:
-------------
1. Multiple hash algorithms (phash, dhash, ahash, whash) for perceptual comparison.
2. Alpha channel hashing for transparent textures.
3. MD5 checksums for exact file duplicate detection.
4. Duplicate analysis and reporting.
5. Parallel processing for speed on large texture packs.
6. Comprehensive metadata collection for each file.
7. Backward compatibility with older map.py and convert.py.

Output Structure:
-----------------
{
  "metadata": {...},                     # General info about run
  "hashes": {"path": "phash"},           # For backward compatibility
  "extended": {                          # Enhanced per-file data
    "path": {
      "algorithms": {"phash": "...", "dhash": "...", ...},
      "md5": "checksum",
      "alpha_hash": "...",               # Optional, only for alpha textures
      "metadata": {...}                  # File size, mode, original format
    }
  },
  "analysis": {...}                      # Duplicate analysis and statistics
}
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from collections import Counter, defaultdict

# ---------------------------------------------------------------------
# DEPENDENCY CHECKING
# ---------------------------------------------------------------------

def check_dependencies():
    """
    Ensure all required Python packages are installed.

    Packages:
      - Pillow: Image loading and manipulation
      - imagehash: Hash computation
      - tqdm: Progress bars for interactive feedback

    Prints missing packages and instructions to install them.
    """
    missing = []

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    try:
        import imagehash
    except ImportError:
        missing.append("imagehash")

    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        for dep in missing:
            print(f"  ❌ {dep}")
        print("\nInstall dependencies with: pip install Pillow imagehash tqdm")
        print("=" * 60)
        return False

    return True

if not check_dependencies():
    sys.exit(1)

from PIL import Image
import imagehash
from tqdm import tqdm

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Hash algorithms available:
# phash: primary perceptual hash (backward compatibility)
# dhash: difference hash
# ahash: average hash
# whash: wavelet hash
HASH_ALGORITHMS = {
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "ahash": imagehash.average_hash,
    "whash": imagehash.whash,
}

DEFAULT_HASH_SIZE = 16             # 16x16 hash for sufficient precision
DEFAULT_NORMALIZE_SIZE = (256, 256)  # Resize images to this size before hashing

# ---------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------

def scan_directory(directory):
    """
    Recursively find all PNG files in a directory and validate them.

    Returns a list of tuples:
      (relative_path, full_path)

    Notes:
      - Skips unreadable files.
      - Validates PNG header for basic correctness.
      - Relative paths used for mapping to SoH texture paths.
    """
    files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.lower().endswith('.png'):
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, directory)

            # Quick PNG validation
            try:
                with open(full_path, 'rb') as f:
                    if f.read(8) == b'\x89PNG\r\n\x1a\n':
                        files.append((rel_path, full_path))
            except OSError:
                # Skip files we cannot read
                continue

    return files


def calculate_md5(filepath):
    """
    Compute the MD5 checksum of a file.

    Used to detect exact duplicates regardless of visual similarity.
    """
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def normalize_image(img, target_size=None, preserve_alpha=False):
    """
    Prepare image for consistent hashing.

    Resizes image and converts to RGB or RGBA if necessary.
    - preserve_alpha=True keeps RGBA mode for separate alpha hashing
    - If image has transparency but preserve_alpha=False, composite on white
    """
    if target_size:
        try:
            img = img.resize(target_size, Image.Resampling.BICUBIC)
        except AttributeError:
            # Older Pillow fallback
            img = img.resize(target_size, Image.BICUBIC)

    # Convert palette images to RGBA
    if img.mode in ('P', 'PA'):
        img = img.convert('RGBA')

    # Convert 1-bit images to grayscale
    if img.mode == '1':
        return img.convert('L')

    # Return grayscale as-is
    if img.mode == 'L':
        return img

    if img.mode == 'RGBA':
        if preserve_alpha:
            return img  # Keep RGBA for alpha hashing
        # Composite onto white background for hashing RGB only
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background

    if img.mode == 'RGB':
        return img

    # Fallback conversion
    return img.convert('RGB')


def compute_alpha_hash(img, hash_size):
    """
    Compute hash of alpha channel only.

    Returns None if image has no alpha channel.
    Useful to detect transparency differences between textures.
    """
    if img.mode != 'RGBA':
        return None

    alpha = img.split()[-1]
    return str(imagehash.phash(alpha, hash_size=hash_size))


def compute_all_hashes(img, hash_size):
    """
    Compute all configured hash algorithms for a normalized image.

    Returns a tuple:
      (results_dict, failures_dict)
    """
    results = {}
    failures = {}

    for name, func in HASH_ALGORITHMS.items():
        try:
            results[name] = str(func(img, hash_size=hash_size))
        except Exception as e:
            failures[name] = str(e)

    return results, failures


def process_single_file(args):
    """
    Process one file to compute all hashes.

    Args tuple:
      (relative_path, full_path, options_dict)

    Returns:
      (success_flag, result_dict)

    Notes:
      - Computes MD5 for exact duplicates.
      - Optionally computes alpha channel hash.
      - Normalizes image before hashing.
      - Returns detailed metadata for downstream analysis.
    """
    rel_path, full_path, options = args

    try:
        with Image.open(full_path) as img:
            # Collect basic file metadata
            file_md5 = calculate_md5(full_path)
            metadata = {
                'original_mode': img.mode,
                'original_size': img.size,
                'original_format': img.format,
                'file_md5': file_md5,
                'file_size': os.path.getsize(full_path)
            }

            # Alpha hash
            alpha_hash = None
            if options.get('alpha_hash', False):
                alpha_hash = compute_alpha_hash(img, options['hash_size'])

            # Normalize image for consistent hashing
            normalized = normalize_image(
                img,
                target_size=options.get('normalize_size'),
                preserve_alpha=options.get('preserve_alpha', False)
            )

            # Compute all hashes
            hashes, failures = compute_all_hashes(normalized, options['hash_size'])

            if not hashes:
                return False, {
                    'file': rel_path,
                    'error': 'All hash algorithms failed',
                    'failures': failures,
                    'metadata': metadata
                }

            # Success
            return True, {
                'file': rel_path,
                'filename': os.path.basename(rel_path),
                'hashes': hashes,
                'alpha_hash': alpha_hash,
                'failures': failures,
                'metadata': metadata,
                'success': True
            }

    except Exception as e:
        return False, {
            'file': rel_path,
            'error': str(e),
            'type': type(e).__name__
        }

# ---------------------------------------------------------------------
# INTERACTIVE WORKFLOW
# ---------------------------------------------------------------------

def collect_user_input():
    """
    Prompt the user for configuration options.

    Returns a dictionary of processing options:
      - directory: folder containing PNG textures
      - format: glide or soh
      - hash_size: size of hashes
      - alpha_hash: whether to compute alpha channel hash
      - preserve_alpha: whether to keep alpha during normalization
      - normalize_size: target size for resizing
      - parallel: use multiprocessing
    """
    print("\n" + "=" * 60)
    print("ENHANCED TEXTURE HASH GENERATOR")
    print("=" * 60)

    # Input folder
    while True:
        directory = input("\n📁 Enter texture directory: ").strip()
        if directory.startswith('~'):
            directory = os.path.expanduser(directory)
        if os.path.exists(directory):
            break
        print(f"❌ Directory not found: {directory}")

    # Texture format
    print("\n🏷️  Texture format:")
    print("  1. Glide/Rice (community texture packs)")
    print("  2. SoH/o2r (game reference textures)")

    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice == '1':
            format_type = 'glide'
            break
        elif choice == '2':
            format_type = 'soh'
            break

    # Processing options
    print("\n⚙️  Processing options (press Enter for defaults):")

    hash_size = input(f"  Hash size [{DEFAULT_HASH_SIZE}]: ").strip()
    hash_size = int(hash_size) if hash_size else DEFAULT_HASH_SIZE

    alpha_hash = input("  Enable alpha channel hashing? (y/N): ").lower().strip() == 'y'
    preserve_alpha = input("  Preserve alpha in preprocessing? (y/N): ").lower().strip() == 'y'

    normalize_size = DEFAULT_NORMALIZE_SIZE
    disable_norm = input("  Disable image resizing? (y/N): ").lower().strip() == 'y'
    if disable_norm:
        normalize_size = None

    parallel = input(f"  Use parallel processing? (Y/n): ").lower().strip() != 'n'

    return {
        'directory': directory,
        'format': format_type,
        'hash_size': hash_size,
        'alpha_hash': alpha_hash,
        'preserve_alpha': preserve_alpha,
        'normalize_size': normalize_size,
        'parallel': parallel
    }

def analyze_results(results):
    """
    Analyze hashing results for duplicates and patterns.

    Returns a dictionary containing:
      - duplicates_by_md5: exact duplicate files
      - duplicates_by_phash: perceptual duplicates
      - filename_conflicts: files with same name
      - statistics: summary counters
    """
    analysis = {
        'duplicates_by_md5': defaultdict(list),
        'duplicates_by_phash': defaultdict(list),
        'filename_conflicts': defaultdict(list),
        'statistics': Counter()
    }

    for rel_path, data in results.items():
        if not data.get('success'):
            continue

        md5 = data['metadata']['file_md5']
        analysis['duplicates_by_md5'][md5].append(rel_path)

        phash = data['hashes'].get('phash')
        if phash:
            analysis['duplicates_by_phash'][phash].append(rel_path)

        filename = os.path.basename(rel_path)
        analysis['filename_conflicts'][filename].append(rel_path)

        if data.get('alpha_hash'):
            analysis['statistics']['alpha_textures'] += 1

    return analysis

# ---------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------

def main():
    """Main interactive workflow for enhanced texture hashing."""
    config = collect_user_input()

    print(f"\n🔍 Scanning {config['directory']} for PNG files...")
    files = scan_directory(config['directory'])
    if not files:
        print("❌ No PNG files found")
        return

    print(f"  Found {len(files)} PNG files")

    options = {
        'hash_size': config['hash_size'],
        'alpha_hash': config['alpha_hash'],
        'preserve_alpha': config['preserve_alpha'],
        'normalize_size': config['normalize_size']
    }

    print("\n🔨 Processing files...")
    results = {}
    errors = []
    stats = Counter()

    if config['parallel'] and len(files) > 10:
        from multiprocessing import Pool
        cpu_count = os.cpu_count() or 4
        with Pool(cpu_count) as pool:
            tasks = [(rel, full, options) for rel, full in files]
            for success, data in tqdm(pool.imap_unordered(process_single_file, tasks),
                                      total=len(tasks),
                                      desc="Hashing"):
                if success:
                    results[data['file']] = data
                    stats['success'] += 1
                else:
                    errors.append(data)
                    stats['failed'] += 1
    else:
        for rel_path, full_path in tqdm(files, desc="Hashing"):
            success, data = process_single_file((rel_path, full_path, options))
            if success:
                results[data['file']] = data
                stats['success'] += 1
            else:
                errors.append(data)
                stats['failed'] += 1

    print("\n📊 Analyzing results...")
    analysis = analyze_results(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{config['format']}_hashes_{timestamp}.json"

    output = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'version': '2.0-enhanced',
            'format': config['format'],
            'directory': config['directory'],
            'total_files': len(files),
            'successful': stats['success'],
            'failed': stats['failed'],
            'processing_options': options
        },
        'hashes': {path: data['hashes']['phash'] for path, data in results.items() if 'phash' in data['hashes']},
        'extended': {path: {'algorithms': data['hashes'],
                            'alpha_hash': data.get('alpha_hash'),
                            'metadata': data['metadata']} for path, data in results.items()},
        'analysis': {
            'duplicates': {
                'exact_files': {md5: paths[:10] for md5, paths in analysis['duplicates_by_md5'].items() if len(paths) > 1},
                'perceptual_matches': {phash: paths[:10] for phash, paths in analysis['duplicates_by_phash'].items() if len(paths) > 1}
            },
            'statistics': {
                'total_files': len(files),
                'successful': stats['success'],
                'alpha_textures': analysis['statistics'].get('alpha_textures', 0),
                'exact_duplicates': sum(1 for paths in analysis['duplicates_by_md5'].values() if len(paths) > 1),
                'perceptual_duplicates': sum(1 for paths in analysis['duplicates_by_phash'].values() if len(paths) > 1)
            }
        },
        'errors': errors[:100]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("HASHING COMPLETE")
    print("=" * 60)
    print(f"✅ Output file: {output_file}")
    print(f"📊 Successfully hashed: {stats['success']}/{len(files)} files")
    dup_count = output['analysis']['statistics']['perceptual_duplicates']
    if dup_count > 0:
        print(f"🔄 Perceptual duplicates detected: {dup_count} sets")
        if config['format'] == 'soh':
            print("   Note: duplicate textures are common in SoH assets.")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run this script for BOTH Glide and SoH texture directories.")
    print("2. Use ENHANCED map.py for improved matching.")
    print("3. Multiple hash algorithms help resolve conflicts.")
    print("4. convert.py will use this hash output for texture conversion.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
