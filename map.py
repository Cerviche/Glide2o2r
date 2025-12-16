#!/usr/bin/env python3
"""
Complete Auto-Adaptive Texture Matcher - Enhanced Documentation
===============================================================

This script performs the automatic matching of Glide/Rice textures to SoH/o2r
textures using perceptual and algorithmic hashing. It adapts dynamically to
available hardware and dataset sizes, aiming for high accuracy while being
efficient with memory and computation.

Key Features:
-------------
1. Hardware-adaptive batch sizes for CPU/GPU
2. Weighted multi-algorithm hash comparison (phash, dhash, ahash, whash)
3. Alpha channel support for transparent textures
4. Duplicate detection and confidence scoring
5. Output in CSV and JSON formats
6. Interactive configuration for matching thresholds

Output:
-------
CSV: Lightweight human-readable match map
JSON: Detailed report with metadata, statistics, and batch configuration
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------
# HARDWARE DETECTION
# ---------------------------------------------------------------------

def detect_hardware():
    """
    Detect available CPU and GPU resources.

    Returns a dictionary with:
      - CPU cores
      - System RAM
      - GPU availability and memory
      - GPU name and cores
    """
    info = {
        'gpu_available': False,
        'gpu_memory_gb': 0,
        'gpu_name': 'None',
        'cpu_cores': os.cpu_count() or 4,
        'system_memory_gb': 8,
    }

    try:
        import cupy as cp
        info['gpu_available'] = True
        try:
            # Attempt to get actual GPU memory and properties
            mem_info = cp.cuda.runtime.memGetInfo()
            info['gpu_memory_gb'] = mem_info[1] / (1024**3)
            info['gpu_memory_free_gb'] = mem_info[0] / (1024**3)
            props = cp.cuda.runtime.getDeviceProperties(0)
            info['gpu_name'] = props['name'].decode('utf-8', 'ignore')
            info['gpu_cores'] = props['multiProcessorCount']
        except Exception:
            # Fallback assumption for unknown GPU
            info['gpu_memory_gb'] = 4
    except ImportError:
        info['gpu_available'] = False

    return info


def get_optimal_batch_sizes(hardware_info, glide_count, soh_count):
    """
    Determine optimal batch sizes for Glide and SoH textures based on
    available hardware and dataset sizes.

    Logic:
    - Prioritize GPU if available, with large batch sizes for high-memory GPUs
    - Fallback to CPU with safe batch sizes based on system RAM
    - Ensure batches are not larger than dataset
    - Minimum batch sizes enforced for safety
    """
    if hardware_info['gpu_available']:
        gpu_mem = hardware_info['gpu_memory_gb']
        if gpu_mem >= 24:
            glide_batch, soh_chunk = 1024, 4096
        elif gpu_mem >= 12:
            glide_batch, soh_chunk = 512, 2048
        elif gpu_mem >= 6:
            glide_batch, soh_chunk = 256, 1024
        elif gpu_mem >= 4:
            glide_batch, soh_chunk = 128, 512
        else:
            glide_batch, soh_chunk = 64, 256
    else:
        sys_mem = hardware_info['system_memory_gb']
        if sys_mem >= 32:
            glide_batch, soh_chunk = 128, 1024
        elif sys_mem >= 16:
            glide_batch, soh_chunk = 64, 512
        else:
            glide_batch, soh_chunk = 32, 256

    # Respect actual dataset size
    glide_batch = min(glide_batch, glide_count)
    soh_chunk = min(soh_chunk, soh_count)

    # Enforce safe minimums
    glide_batch = max(16, glide_batch)
    soh_chunk = max(64, soh_chunk)

    return glide_batch, soh_chunk

# ---------------------------------------------------------------------
# HASH PROCESSING UTILITIES
# ---------------------------------------------------------------------

def hex_to_binary_array(hex_str, bits=256):
    """
    Convert a hexadecimal hash string to a fixed-size binary numpy array.

    Returns an array of 0s and 1s representing each bit.
    """
    if not hex_str:
        return np.zeros(bits, dtype=np.uint8)
    try:
        hash_int = int(hex_str, 16)
        binary = np.zeros(bits, dtype=np.uint8)
        for i in range(bits):
            binary[i] = (hash_int >> (bits - 1 - i)) & 1
        return binary
    except ValueError:
        return np.zeros(bits, dtype=np.uint8)


def load_and_prepare_hashes(hash_file):
    """
    Load a hash JSON file and convert all hashes into numpy arrays for matching.

    Supports both legacy (phash only) and extended (multi-algorithm) formats.

    Returns:
      - paths: list of texture paths
      - algorithms: dictionary of numpy arrays for each hash type
      - alpha_array: numpy array for alpha hashes
    """
    print(f"Loading {hash_file.name}...")
    with open(hash_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    paths = []
    algorithms = { 'phash': [], 'dhash': [], 'ahash': [], 'whash': [] }
    alpha_list = []

    if 'extended' in data:
        hash_items = data['extended'].items()
    elif 'hashes' in data:
        # Legacy format fallback
        hash_items = [(path, {'algorithms': {'phash': hash_val}})
                      for path, hash_val in data['hashes'].items()]
    else:
        raise ValueError("Invalid hash file format")

    for path, item_data in tqdm(hash_items, desc="Processing", unit="textures"):
        paths.append(path)
        algs = item_data.get('algorithms', {})
        for alg in algorithms:
            hex_str = algs.get(alg, '')
            algorithms[alg].append(hex_to_binary_array(hex_str, 256))

        alpha_hex = item_data.get('alpha_hash', '')
        alpha_list.append(hex_to_binary_array(alpha_hex, 256) if alpha_hex
                          else np.zeros(256, dtype=np.uint8))

    for alg in algorithms:
        algorithms[alg] = np.array(algorithms[alg], dtype=np.uint8)
    alpha_array = np.array(alpha_list, dtype=np.uint8)

    return paths, algorithms, alpha_array

# ---------------------------------------------------------------------
# MATCHING ENGINE
# ---------------------------------------------------------------------

class MatchingEngine:
    """
    Optimized matching engine with optional GPU acceleration.

    Uses weighted combination of multiple hash algorithms.
    """
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.cp = None
        if use_gpu:
            import cupy as cp
            self.cp = cp

        # Weighting for each hash type in distance calculation
        self.weights = {'phash': 0.5, 'dhash': 0.2, 'ahash': 0.15, 'whash': 0.15}

    def compute_distances(self, glide_batch, soh_chunk):
        """Compute distances between Glide batch and SoH chunk."""
        if self.use_gpu and self.cp:
            return self._compute_gpu(glide_batch, soh_chunk)
        else:
            return self._compute_cpu(glide_batch, soh_chunk)

    def _compute_gpu(self, glide_batch, soh_chunk):
        """GPU-based distance calculation using CuPy arrays."""
        batch_size = glide_batch['phash'].shape[0]
        chunk_size = soh_chunk['phash'].shape[0]
        distances = self.cp.zeros((batch_size, chunk_size), dtype=self.cp.float32)

        for alg, weight in self.weights.items():
            g_arr = self.cp.array(glide_batch[alg])[:, self.cp.newaxis, :]
            s_arr = self.cp.array(soh_chunk[alg])[self.cp.newaxis, :, :]
            distances += self.cp.sum(g_arr != s_arr, axis=2) * weight

        return self.cp.asnumpy(distances)

    def _compute_cpu(self, glide_batch, soh_chunk):
        """CPU-based distance calculation using NumPy."""
        batch_size = glide_batch['phash'].shape[0]
        chunk_size = soh_chunk['phash'].shape[0]
        distances = np.zeros((batch_size, chunk_size), dtype=np.float32)

        for alg, weight in self.weights.items():
            g_arr = glide_batch[alg][:, np.newaxis, :]
            s_arr = soh_chunk[alg][np.newaxis, :, :]
            distances += np.sum(g_arr != s_arr, axis=2) * weight

        return distances

# ---------------------------------------------------------------------
# MATCHING WORKFLOW
# ---------------------------------------------------------------------

def find_matches(engine, glide_paths, glide_arrays, glide_alpha,
                 soh_paths, soh_arrays, soh_alpha,
                 glide_batch_size, soh_chunk_size, max_distance=10.0):
    """
    Core matching routine:
    - Process Glide textures in batches
    - Compare against SoH textures in chunks
    - Track best match for each Glide texture
    - Calculate confidence, alpha matching, and duplicates
    """
    total_glide = len(glide_paths)
    total_soh = len(soh_paths)

    print(f"\n🔍 Starting matching: {total_glide} Glide vs {total_soh} SoH")
    print(f"   Batch size: Glide={glide_batch_size}, SoH={soh_chunk_size}")
    print(f"   Max distance: {max_distance}")

    matches = []
    stats = Counter()

    # Glide batch processing
    for glide_start in tqdm(range(0, total_glide, glide_batch_size),
                           desc="Processing", unit="batch"):
        glide_end = min(glide_start + glide_batch_size, total_glide)
        glide_slice = slice(glide_start, glide_end)
        glide_batch = {alg: arr[glide_slice] for alg, arr in glide_arrays.items()}
        glide_alpha_batch = glide_alpha[glide_slice]
        batch_paths = glide_paths[glide_start:glide_end]

        batch_best_dist = np.full(len(batch_paths), np.inf)
        batch_best_idx = np.full(len(batch_paths), -1, dtype=int)
        batch_best_alg_dists = [{} for _ in range(len(batch_paths))]
        batch_has_alpha = [False] * len(batch_paths)

        for soh_start in range(0, total_soh, soh_chunk_size):
            soh_end = min(soh_start + soh_chunk_size, total_soh)
            soh_slice = slice(soh_start, soh_end)
            soh_chunk = {alg: arr[soh_slice] for alg, arr in soh_arrays.items()}
            soh_alpha_chunk = soh_alpha[soh_slice]

            distances = engine.compute_distances(glide_batch, soh_chunk)

            # Evaluate best match per Glide texture
            for i in range(distances.shape[0]):
                chunk_best = np.argmin(distances[i])
                chunk_dist = distances[i, chunk_best]

                if chunk_dist < batch_best_dist[i]:
                    batch_best_dist[i] = chunk_dist
                    batch_best_idx[i] = soh_start + chunk_best

                    alg_dists = {}
                    for alg in engine.weights:
                        g_bits = glide_batch[alg][i]
                        s_bits = soh_chunk[alg][chunk_best]
                        alg_dists[alg] = int(np.sum(g_bits != s_bits))

                    alpha_diff = int(np.sum(glide_alpha_batch[i] != soh_alpha_chunk[chunk_best]))
                    if alpha_diff <= 5:
                        alg_dists['alpha'] = alpha_diff
                        batch_has_alpha[i] = True

                    batch_best_alg_dists[i] = alg_dists

        # Record matches for the batch
        for i, glide_idx in enumerate(range(glide_start, glide_end)):
            if batch_best_dist[i] <= max_distance and batch_best_idx[i] >= 0:
                alg_dists = batch_best_alg_dists[i]
                if alg_dists:
                    normalized = [d / 256 for d in alg_dists.values()]
                    variance = np.var(normalized) if len(normalized) > 1 else 0
                    confidence = 1.0 / (1.0 + 10.0 * variance)
                else:
                    confidence = 0.5

                match = {
                    'glide_path': batch_paths[i],
                    'glide_filename': os.path.basename(batch_paths[i]),
                    'soh_primary_path': soh_paths[batch_best_idx[i]],
                    'weighted_distance': float(batch_best_dist[i]),
                    'confidence': float(confidence),
                    'algorithm_distances': alg_dists,
                    'has_alpha_match': batch_has_alpha[i],
                    'duplicate_count': 1
                }
                matches.append(match)
                stats['matched'] += 1
            else:
                stats['unmatched'] += 1

    # Duplicate detection
    print("\n🔍 Detecting duplicates...")
    path_to_matches = defaultdict(list)
    for match in matches:
        path_to_matches[match['soh_primary_path']].append(match)

    for soh_path, match_list in path_to_matches.items():
        dup_count = len(match_list)
        for match in match_list:
            match['duplicate_count'] = dup_count
            match['soh_all_paths'] = '|'.join(m['soh_primary_path'] for m in match_list)

        if dup_count > 1:
            stats['duplicate_sets'] = stats.get('duplicate_sets', 0) + 1
            stats['additional_copies'] = stats.get('additional_copies', 0) + (dup_count - 1)

    print(f"  Found {stats.get('duplicate_sets', 0)} duplicate sets")
    return matches, dict(stats)

# ---------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------

def main():
    """
    Main execution:
    - Detect hardware
    - Find latest Glide/SoH hash files
    - Calculate batch sizes
    - Load hash arrays
    - Initialize matching engine
    - Run matching
    - Save CSV and JSON outputs
    """
    print("\n" + "=" * 70)
    print("AUTO-ADAPTIVE TEXTURE MATCHING - WORKING VERSION")
    print("=" * 70)

    # Hardware detection
    print("\n🔍 Detecting hardware...")
    hardware = detect_hardware()

    # Hash files discovery
    print("\n📁 Finding hash files...")
    glide_files = sorted(Path('.').glob('glide_hashes_*.json'))
    soh_files = sorted(Path('.').glob('soh_hashes_*.json'))

    if not glide_files:
        print("❌ No Glide hash files found")
        return
    if not soh_files:
        print("❌ No SoH hash files found")
        return

    glide_file = glide_files[-1]
    soh_file = soh_files[-1]

    print(f"   Glide: {glide_file.name}")
    print(f"   SoH: {soh_file.name}")

    # Dataset sizes
    print("\n📊 Checking dataset sizes...")
    with open(glide_file, 'r') as f:
        glide_data = json.load(f)
        glide_count = len(glide_data.get('extended', glide_data.get('hashes', {})))

    with open(soh_file, 'r') as f:
        soh_data = json.load(f)
        soh_count = len(soh_data.get('extended', soh_data.get('hashes', {})))

    print(f"   Glide textures: {glide_count:,}")
    print(f"   SoH textures: {soh_count:,}")
    print(f"   Total comparisons: {glide_count * soh_count:,}")

    # Batch size calculation
    print("\n⚙️  Calculating optimal settings...")
    glide_batch, soh_chunk = get_optimal_batch_sizes(hardware, glide_count, soh_count)

    print(f"\n📊 HARDWARE:")
    print(f"   CPU Cores: {hardware['cpu_cores']}")
    print(f"   System RAM: {hardware['system_memory_gb']:.1f} GB")
    if hardware['gpu_available']:
        print(f"   ✅ GPU: {hardware['gpu_name']}")
        print(f"   GPU Memory: {hardware['gpu_memory_gb']:.1f} GB")
        mode = "GPU"
    else:
        print(f"   ⚠ GPU: Not available")
        mode = "CPU"

    print(f"\n🎯 OPTIMIZATION:")
    print(f"   Mode: {mode}")
    print(f"   Glide batch: {glide_batch}")
    print(f"   SoH chunk: {soh_chunk}")

    # Matching threshold
    print("\n⚙️  Matching configuration:")
    try:
        max_dist = float(input("Max weighted distance [10.0]: ").strip() or "10.0")
    except:
        max_dist = 10.0

    # Load hash arrays
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    print("\n📥 Loading Glide hashes...")
    glide_paths, glide_arrays, glide_alpha = load_and_prepare_hashes(glide_file)

    print("\n📥 Loading SoH hashes...")
    soh_paths, soh_arrays, soh_alpha = load_and_prepare_hashes(soh_file)

    # Initialize matching engine
    print(f"\n⚡ Creating {mode} matching engine...")
    engine = MatchingEngine(use_gpu=hardware['gpu_available'])

    # Perform matching
    print("\n" + "=" * 70)
    print("MATCHING TEXTURES")
    print("=" * 70)

    matches, stats = find_matches(
        engine=engine,
        glide_paths=glide_paths,
        glide_arrays=glide_arrays,
        glide_alpha=glide_alpha,
        soh_paths=soh_paths,
        soh_arrays=soh_arrays,
        soh_alpha=soh_alpha,
        glide_batch_size=glide_batch,
        soh_chunk_size=soh_chunk,
        max_distance=max_dist
    )

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"texture_map_{timestamp}.csv"
    json_file = f"texture_map_{timestamp}.json"

    # CSV export
    print(f"\n💾 Saving {len(matches):,} matches to {csv_file}")
    csv_fields = [
        'glide_path', 'glide_filename', 'soh_primary_path', 'soh_all_paths',
        'weighted_distance', 'confidence', 'algorithm_distances',
        'duplicate_count', 'has_alpha_match'
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for match in matches:
            alg_str = '|'.join(f"{k}:{v}" for k, v in match['algorithm_distances'].items())
            row = {
                'glide_path': match['glide_path'],
                'glide_filename': match['glide_filename'],
                'soh_primary_path': match['soh_primary_path'],
                'soh_all_paths': match.get('soh_all_paths', match['glide_filename']),
                'weighted_distance': match['weighted_distance'],
                'confidence': match['confidence'],
                'algorithm_distances': alg_str,
                'duplicate_count': match.get('duplicate_count', 1),
                'has_alpha_match': match['has_alpha_match']
            }
            writer.writerow(row)

    # JSON export
    print(f"💾 Saving JSON report to {json_file}")
    report = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'glide_file': glide_file.name,
            'soh_file': soh_file.name,
            'max_distance': max_dist,
            'hardware': hardware,
            'batch_sizes': {'glide': glide_batch, 'soh': soh_chunk}
        },
        'statistics': {
            'glide_textures': glide_count,
            'soh_textures': soh_count,
            'matches_found': len(matches),
            'unmatched': stats.get('unmatched', 0),
            'match_rate': len(matches) / glide_count if glide_count > 0 else 0,
            'duplicate_sets': stats.get('duplicate_sets', 0),
            'additional_copies': stats.get('additional_copies', 0)
        }
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("MATCHING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 RESULTS:")
    print(f"   Matches found: {len(matches):,}")
    print(f"   Match rate: {len(matches)/glide_count*100:.1f}%")

    if matches:
        avg_dist = np.mean([m['weighted_distance'] for m in matches])
        avg_conf = np.mean([m['confidence'] for m in matches])
        print(f"   Average distance: {avg_dist:.2f}")
        print(f"   Average confidence: {avg_conf:.3f}")

    if stats.get('duplicate_sets', 0) > 0:
        print(f"\n🔄 Duplicates:")
        print(f"   Duplicate sets: {stats['duplicate_sets']}")
        print(f"   Additional copies needed: {stats.get('additional_copies', 0)}")

    print(f"\n💾 Output files:")
    print(f"   CSV map: {csv_file}")
    print(f"   JSON report: {json_file}")

    print(f"\n➡️  Next: Run convert.py with {csv_file}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
