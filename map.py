#!/usr/bin/env python3
"""
Texture Mapping Generator for SoH Texture Pipeline
==================================================
Creates CSV maps between texture formats using cached hashes.

Purpose:
--------
This script is the second step in a 3-step texture conversion pipeline:
1. hash.py: Calculate perceptual hashes for all textures (ONE-TIME, SLOW)
2. map.py: Create mapping between formats using cached hashes (FAST, REPEATABLE)
3. convert.py: Convert texture packs using generated maps

This script loads pre-computed hashes from hash.py and creates mappings
between Glide and SoH format textures at user-specified Hamming distance
thresholds. It can be run multiple times with different thresholds without
re-hashing textures.

Key Features:
-------------
- Fast matching using cached hashes (no re-hashing needed)
- Configurable Hamming distance threshold (0-10)
- Handles duplicate textures (same content, multiple locations)
- Generates comprehensive CSV and JSON reports
- Never modifies original texture files (read-only)

Usage:
------
python map.py

Dependencies:
------------
- tqdm: For progress bars
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Third-party import for progress bars
from tqdm import tqdm


def find_hash_files():
    """
    Find hash files created by hash.py.

    Searches for files matching patterns:
    - glide_hashes_*.json (Glide format hashes)
    - soh_hashes_*.json (SoH format hashes)

    Returns:
        tuple: (glide_files, soh_files) - lists of Path objects
    """
    glide_files = list(Path('.').glob('glide_hashes_*.json'))
    soh_files = list(Path('.').glob('soh_hashes_*.json'))

    return glide_files, soh_files


def load_hash_file(hash_file):
    """
    Load hash data from JSON file created by hash.py.

    Args:
        hash_file (Path): Path to hash JSON file

    Returns:
        tuple: (metadata, hashes_dict, file_list)
            metadata: Dictionary with file metadata
            hashes_dict: Dictionary of rel_path -> hash_string
            file_list: List of all files with successful hashes
    """
    print(f"Loading hash file: {hash_file.name}")

    with open(hash_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    hashes = data.get('hashes', {})
    file_list = data.get('file_list', list(hashes.keys()))

    print(f"  Format: {metadata.get('format', 'unknown').upper()}")
    print(f"  Textures: {len(file_list)}")
    print(f"  Directory: {metadata.get('directory', 'unknown')}")

    return metadata, hashes, file_list


def create_texture_map():
    """
    Main function to create texture mapping between formats.

    This function:
    1. Loads pre-computed Glide and SoH hashes
    2. Finds matches at user-specified Hamming distance threshold
    3. Handles duplicate textures (same hash, multiple paths)
    4. Generates CSV map and JSON debug info
    5. Never modifies original files (read-only)

    Returns:
        str or None: Path to created CSV map, or None if failed
    """
    print("=" * 60)
    print("TEXTURE MAPPING GENERATOR")
    print("Step 2 of 3 in texture conversion pipeline")
    print("=" * 60)
    print("This step creates mappings using cached hashes (FAST)")
    print("Can be rerun with different thresholds without re-hashing")
    print("=" * 60)

    # STEP 1: Find available hash files
    print("\n--- Step 1: Select Hash Files ---")
    glide_hash_files, soh_hash_files = find_hash_files()

    if not glide_hash_files:
        print("No Glide hash files found.")
        print("Please run hash.py on your Glide directory first.")
        return None

    if not soh_hash_files:
        print("No SoH hash files found.")
        print("Please run hash.py on your SoH directory first.")
        return None

    # Let user select Glide hash file
    print(f"\nFound {len(glide_hash_files)} Glide hash files:")
    for i, hash_file in enumerate(sorted(glide_hash_files, reverse=True)[:5], 1):
        size_kb = hash_file.stat().st_size / 1024
        print(f"  {i}. {hash_file.name} ({size_kb:.0f} KB)")

    if len(glide_hash_files) > 5:
        print(f"  ... and {len(glide_hash_files) - 5} more")

    glide_choice = input(f"\nSelect Glide hash file (1-{min(5, len(glide_hash_files))}) or enter filename: ").strip()

    if glide_choice.isdigit() and 1 <= int(glide_choice) <= len(glide_hash_files):
        selected_glide = glide_hash_files[int(glide_choice) - 1]
    else:
        selected_glide = Path(glide_choice)
        if not selected_glide.exists():
            print(f"Error: Glide hash file not found: {glide_choice}")
            return None

    # Let user select SoH hash file
    print(f"\nFound {len(soh_hash_files)} SoH hash files:")
    for i, hash_file in enumerate(sorted(soh_hash_files, reverse=True)[:5], 1):
        size_kb = hash_file.stat().st_size / 1024
        print(f"  {i}. {hash_file.name} ({size_kb:.0f} KB)")

    if len(soh_hash_files) > 5:
        print(f"  ... and {len(soh_hash_files) - 5} more")

    soh_choice = input(f"\nSelect SoH hash file (1-{min(5, len(soh_hash_files))}) or enter filename: ").strip()

    if soh_choice.isdigit() and 1 <= int(soh_choice) <= len(soh_hash_files):
        selected_soh = soh_hash_files[int(soh_choice) - 1]
    else:
        selected_soh = Path(soh_choice)
        if not selected_soh.exists():
            print(f"Error: SoH hash file not found: {soh_choice}")
            return None

    # STEP 2: Get Hamming distance threshold
    print("\n--- Step 2: Matching Threshold ---")
    print("Hamming distance measures similarity between perceptual hashes.")
    print("Lower = more similar, 0 = identical")
    print("\nRecommended thresholds:")
    print("  0-1: Perfect/Very High confidence (recommended)")
    print("  2: High confidence (balanced)")
    print("  3: Moderate confidence (more matches, some risk)")
    print("  4+: Low confidence (high risk of false matches)")

    while True:
        try:
            threshold_str = input("\nMaximum Hamming distance (0-10, recommended: 2): ").strip()
            max_hamming = int(threshold_str) if threshold_str else 2

            if 0 <= max_hamming <= 10:
                break
            else:
                print("Please enter a value between 0 and 10")
        except ValueError:
            print("Please enter a valid number")

    # STEP 3: Load hash data
    print("\n--- Step 3: Loading Hash Data ---")

    print("Loading Glide hashes...")
    glide_metadata, glide_hashes, glide_files = load_hash_file(selected_glide)

    print("\nLoading SoH hashes...")
    soh_metadata, soh_hashes, soh_files = load_hash_file(selected_soh)

    # STEP 4: Prepare for matching
    print("\n--- Step 4: Preparing for Matching ---")

    # Create reverse lookup: hash -> list of SoH paths (for duplicates)
    soh_hash_to_paths = defaultdict(list)
    for soh_path, soh_hash in soh_hashes.items():
        soh_hash_to_paths[soh_hash].append(soh_path)

    # Report on SoH duplicates
    duplicate_stats = Counter()
    for soh_hash, paths in soh_hash_to_paths.items():
        duplicate_stats[len(paths)] += 1

    unique_soh_hashes = len(soh_hash_to_paths)
    total_soh_textures = len(soh_files)
    duplicate_textures = sum((count-1) * freq for count, freq in duplicate_stats.items())

    print(f"\nSoH texture analysis:")
    print(f"  Total textures: {total_soh_textures}")
    print(f"  Unique hashes: {unique_soh_hashes}")
    print(f"  Duplicate textures: {duplicate_textures}")

    if duplicate_stats:
        print(f"  Duplicate distribution:")
        for count in sorted(duplicate_stats.keys()):
            if count > 1:
                freq = duplicate_stats[count]
                print(f"    Hash appears {count} times: {freq} hashes")

    # STEP 5: Find matches
    print(f"\n--- Step 5: Finding Matches (Hamming ≤ {max_hamming}) ---")
    print(f"Matching {len(glide_files)} Glide textures against {unique_soh_hashes} unique SoH hashes...")

    matches = []  # List of match dictionaries
    hamming_distribution = Counter()  # Track distribution of match distances

    # Convert hash strings back to imagehash objects for comparison
    # Note: We need to import imagehash here for Hamming distance calculation
    import imagehash

    # Progress bar for matching
    for glide_path in tqdm(glide_files, desc="Matching", unit="textures"):
        glide_hash_str = glide_hashes.get(glide_path)
        if not glide_hash_str:
            continue  # Skip if hash missing

        # Convert string back to ImageHash object
        glide_hash = imagehash.hex_to_hash(glide_hash_str)

        # Find best matching SoH texture
        best_soh_hash = None
        best_distance = max_hamming + 1  # Start above threshold
        best_soh_paths = []

        for soh_hash_str, soh_paths in soh_hash_to_paths.items():
            soh_hash = imagehash.hex_to_hash(soh_hash_str)
            distance = glide_hash - soh_hash  # Hamming distance

            if distance < best_distance:
                best_distance = distance
                best_soh_hash = soh_hash_str
                best_soh_paths = soh_paths
            elif distance == best_distance and distance <= max_hamming:
                # Multiple matches at same distance - keep all paths
                best_soh_paths.extend(soh_paths)

        # If we found a match within threshold
        if best_distance <= max_hamming and best_soh_paths:
            # Remove duplicates from paths list
            unique_soh_paths = list(dict.fromkeys(best_soh_paths))

            match_info = {
                'glide_filename': glide_path,
                'glide_hash': glide_hash_str,
                'soh_primary_path': unique_soh_paths[0],
                'soh_all_paths': '|'.join(unique_soh_paths),
                'hamming_distance': best_distance,
                'duplicate_count': len(unique_soh_paths)
            }
            matches.append(match_info)
            hamming_distribution[best_distance] += 1

    # STEP 6: Generate output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        'csv_output': f"texture_map_{timestamp}.csv",
        'json_output': f"texture_map_{timestamp}.json",
        'max_hamming': max_hamming,
        'glide_hash_file': selected_glide.name,
        'soh_hash_file': selected_soh.name
    }

    print(f"\n--- Step 6: Saving Results ---")
    print(f"Saving {len(matches)} matches to {config['csv_output']}")

    # Save CSV map
    with open(config['csv_output'], 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'glide_filename',
            'glide_hash',
            'soh_primary_path',
            'soh_all_paths',
            'hamming_distance',
            'duplicate_count'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)

    # Save JSON debug info
    debug_info = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'max_hamming': max_hamming,
            'glide_hash_file': selected_glide.name,
            'soh_hash_file': selected_soh.name,
            'glide_directory': glide_metadata.get('directory'),
            'soh_directory': soh_metadata.get('directory')
        },
        'statistics': {
            'glide_textures': len(glide_files),
            'soh_textures': total_soh_textures,
            'soh_unique_hashes': unique_soh_hashes,
            'soh_duplicate_textures': duplicate_textures,
            'matches_found': len(matches),
            'match_rate': f"{(len(matches)/len(glide_files))*100:.1f}%" if glide_files else "0%",
            'hamming_distribution': {str(k): v for k, v in dict(hamming_distribution).items()},
            'duplicate_distribution': {str(k): v for k, v in dict(duplicate_stats).items()}
        },
        'sample_matches': matches[:50] if matches else []
    }

    with open(config['json_output'], 'w', encoding='utf-8') as jsonfile:
        json.dump(debug_info, jsonfile, indent=2, default=str)

    # STEP 7: Print summary
    print("\n" + "=" * 60)
    print("MAPPING COMPLETE")
    print("=" * 60)
    print(f"Glide textures: {len(glide_files)}")
    print(f"SoH textures: {total_soh_textures} ({unique_soh_hashes} unique, {duplicate_textures} duplicates)")
    print(f"Matches found: {len(matches)} (Hamming ≤ {max_hamming})")
    print(f"Match rate: {debug_info['statistics']['match_rate']}")

    print(f"\nMatch quality distribution:")
    for distance in sorted(hamming_distribution.keys()):
        count = hamming_distribution[distance]
        percentage = (count / len(matches)) * 100 if matches else 0
        print(f"  Hamming {distance}: {count} matches ({percentage:.1f}%)")

    # Calculate average duplicates
    if matches:
        total_duplicates = sum(match['duplicate_count'] for match in matches)
        avg_duplicates = total_duplicates / len(matches)
        print(f"\nAverage destinations per match: {avg_duplicates:.2f}")

    print(f"\nOutput files:")
    print(f"  CSV map: {config['csv_output']}")
    print(f"  JSON debug: {config['json_output']}")

    print(f"\nNEXT STEP:")
    print(f"Run convert.py to use this map for texture pack conversion")

    return config['csv_output']


if __name__ == "__main__":
    """
    Main entry point with error handling.
    """
    try:
        create_texture_map()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
